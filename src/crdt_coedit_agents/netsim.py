"""A deterministic, adversarial network for testing convergence.

Real networks reorder, duplicate, delay and partition. A CRDT is supposed not to care, so the test
harness makes all four happen on purpose, driven by a seeded RNG so a failing run can be replayed
exactly.

Delivery model: every broadcast operation becomes one in-flight message per other replica, with a
random delay in rounds. :meth:`NetworkSimulator.step` advances one round and delivers every due
message whose sender and receiver are on the same side of the current partition; the rest stay in
flight. Messages are never lost — a partition holds them until it heals, which is what a real
reconnect-and-catch-up does.
"""

from __future__ import annotations

import random
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field

from crdt_coedit_agents.ops.model import SignedOp
from crdt_coedit_agents.replica import Replica

__all__ = ["NetworkProfile", "NetworkSimulator", "SimulationStats"]


@dataclass(frozen=True, slots=True)
class NetworkProfile:
    """How unpleasant the network is. All probabilities are per message."""

    duplicate_probability: float = 0.15
    max_delay_rounds: int = 4
    reorder: bool = True

    def __post_init__(self) -> None:
        if not 0.0 <= self.duplicate_probability <= 1.0:
            raise ValueError("duplicate_probability must be a probability")
        if self.max_delay_rounds < 0:
            raise ValueError("max_delay_rounds must be non-negative")


@dataclass(slots=True)
class _Message:
    due_round: int
    sender: str
    target: str
    ops: tuple[SignedOp, ...]


@dataclass(slots=True)
class SimulationStats:
    sent: int = 0
    delivered: int = 0
    duplicated: int = 0
    rounds: int = 0
    held_by_partition: int = 0
    partitions: list[tuple[str, ...]] = field(default_factory=list)


class NetworkSimulator:
    """Moves operations between in-memory replicas under an adversarial delivery schedule."""

    def __init__(
        self,
        replicas: Mapping[str, Replica],
        *,
        seed: int = 0,
        profile: NetworkProfile | None = None,
    ) -> None:
        if len(replicas) < 2:
            raise ValueError("a simulation needs at least two replicas")
        self._replicas = dict(replicas)
        self._profile = profile if profile is not None else NetworkProfile()
        self._rng = random.Random(seed)  # noqa: S311  # nosec B311 - simulation, not cryptography
        self._inflight: list[_Message] = []
        self._round = 0
        self._isolated: frozenset[str] = frozenset()
        self.stats = SimulationStats()

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._replicas))

    @property
    def inflight(self) -> int:
        return len(self._inflight)

    def replica(self, name: str) -> Replica:
        return self._replicas[name]

    # ------------------------------------------------------------- partitions

    def partition(self, isolated: Iterable[str]) -> None:
        """Cut ``isolated`` off from the rest. Messages across the cut wait instead of vanishing."""
        names = frozenset(isolated)
        unknown = names - self._replicas.keys()
        if unknown:
            raise KeyError(f"unknown replicas: {', '.join(sorted(unknown))}")
        self._isolated = names
        self.stats.partitions.append(tuple(sorted(names)))

    def heal(self) -> None:
        self._isolated = frozenset()

    def _reachable(self, sender: str, target: str) -> bool:
        return (sender in self._isolated) == (target in self._isolated)

    # -------------------------------------------------------------- transport

    def broadcast(self, sender: str, ops: Sequence[SignedOp]) -> None:
        """Queue ``ops`` for every other replica, with random delays and duplicates."""
        if sender not in self._replicas:
            raise KeyError(f"unknown replica {sender}")
        if not ops:
            return
        batch = tuple(ops)
        for target in self._replicas:
            if target == sender:
                continue
            copies = 1
            if self._rng.random() < self._profile.duplicate_probability:
                copies = 2
                self.stats.duplicated += 1
            for _ in range(copies):
                delay = self._rng.randint(0, self._profile.max_delay_rounds)
                self._inflight.append(_Message(self._round + delay, sender, target, batch))
                self.stats.sent += 1

    def step(self) -> int:
        """Advance one round, deliver every due reachable message, and count them."""
        self._round += 1
        self.stats.rounds += 1
        due: list[_Message] = []
        held: list[_Message] = []
        for message in self._inflight:
            deliverable = self._reachable(message.sender, message.target)
            if message.due_round <= self._round and deliverable:
                due.append(message)
            else:
                held.append(message)
                if message.due_round <= self._round:
                    self.stats.held_by_partition += 1
        self._inflight = held
        if self._profile.reorder:
            self._rng.shuffle(due)
        for message in due:
            self._replicas[message.target].receive(message.ops)
            self.stats.delivered += 1
        return len(due)

    def settle(self, *, max_rounds: int = 10_000) -> int:
        """Run rounds until nothing deliverable is left. Returns the number of rounds used."""
        rounds = 0
        while rounds < max_rounds:
            deliverable = any(
                self._reachable(message.sender, message.target) for message in self._inflight
            )
            if not deliverable:
                return rounds
            self.step()
            rounds += 1
        raise RuntimeError("network did not settle; a message is stuck in flight")

    # ------------------------------------------------------------- convergence

    def state_hashes(self) -> dict[str, str]:
        return {name: replica.state_hash() for name, replica in sorted(self._replicas.items())}

    def converged(self) -> bool:
        return len(set(self.state_hashes().values())) == 1

    def views(self) -> dict[str, str]:
        return {name: replica.view().text for name, replica in sorted(self._replicas.items())}
