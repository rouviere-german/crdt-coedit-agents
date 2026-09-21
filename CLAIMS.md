# Claims and evidence

Every claim this repository makes, and the test, benchmark or CI job that backs it.
`scripts/check_baseline.py` fails if any reference here does not resolve to a real test function,
file or CI job, so this table cannot rot silently.

Numbers in the README Results section come from `bench/bench_baseline.py` and `bench/bench_crdt.py`
run on the machine named there; they are not estimates and they are not from CI.

## The primary pattern: conflict-free replication

| # | Claim | Evidence |
|---|---|---|
| C1 | Merging is commutative, associative and idempotent for the text sequence, the field map and the tag set | `tests/property/test_algebra.py::test_text_join_is_commutative_associative_and_idempotent`, `tests/property/test_algebra.py::test_field_join_is_commutative_associative_and_idempotent`, `tests/property/test_algebra.py::test_tag_join_is_commutative_associative_and_idempotent` |
| C2 | The same laws hold for the whole document, and merging never mutates either side | `tests/property/test_algebra.py::test_document_join_is_commutative`, `tests/property/test_algebra.py::test_document_join_is_associative`, `tests/property/test_algebra.py::test_document_join_is_idempotent`, `tests/property/test_algebra.py::test_joining_does_not_mutate_either_side` |
| C3 | Replicas that saw different subsets reach one state hash and one text, in any merge order | `tests/property/test_algebra.py::test_the_join_of_all_three_is_what_every_replica_reaches` |
| C4 | Replicas converge under reordering, duplication and delay | `tests/property/test_convergence.py::test_replicas_converge_under_reordering_duplication_and_delay`, `tests/property/test_convergence.py::test_a_duplicate_heavy_network_still_converges` |
| C5 | Replicas converge across a partition that heals | `tests/property/test_convergence.py::test_replicas_converge_across_a_partition_that_heals` |
| C6 | 5 replicas, 10 000 operations, random delivery, duplication and a partition: one state hash, one view | `tests/property/test_convergence.py::test_five_replicas_converge_over_ten_thousand_operations` |
| C7 | Operations may arrive in any order; a node whose parent has not arrived is invisible until it does | `tests/unit/test_replica.py::test_operations_can_arrive_in_any_order_and_still_converge`, `tests/unit/test_fugue.py::test_a_node_whose_parent_has_not_arrived_is_invisible_until_it_does` |
| C8 | Re-delivering an operation changes nothing and is reported as a duplicate | `tests/unit/test_replica.py::test_receiving_the_same_operation_twice_is_a_duplicate_not_an_error`, `tests/unit/test_replica.py::test_applying_an_operation_twice_reports_no_change` |
| C9 | The adversarial network really does reorder, duplicate, delay and hold messages across a partition | `tests/unit/test_snapshot_netsim.py::test_delays_duplicates_and_reordering_are_applied`, `tests/unit/test_snapshot_netsim.py::test_a_partition_holds_messages_until_it_heals` |

## Fugue: non-interleaving and placement

| # | Claim | Evidence |
|---|---|---|
| C10 | Concurrent runs typed left-to-right do not interleave | `tests/property/test_fugue.py::test_concurrent_forward_runs_do_not_interleave` |
| C11 | Concurrent runs typed right-to-left do not interleave either (this is where RGA fails) | `tests/property/test_fugue.py::test_concurrent_backward_runs_do_not_interleave` |
| C12 | A forward run and a backward run at the same anchor do not interleave; neither do three concurrent runs | `tests/property/test_fugue.py::test_a_forward_run_and_a_backward_run_do_not_interleave`, `tests/property/test_fugue.py::test_three_concurrent_runs_stay_in_three_blocks` |
| C13 | A local insert lands immediately after its anchor | `tests/property/test_fugue.py::test_a_local_insert_lands_immediately_after_its_anchor`, `tests/unit/test_fugue.py::test_inserting_between_two_elements_becomes_a_left_child_of_the_successor` |
| C14 | Deleting commutes with a concurrent insert and is idempotent | `tests/property/test_fugue.py::test_deleting_is_idempotent_and_commutes_with_a_concurrent_insert`, `tests/unit/test_fugue.py::test_deleting_hides_an_element_without_moving_the_others` |
| C15 | Sibling order is right-ascending and left-descending, so the structure mirrors under reversal | `tests/unit/test_fugue.py::test_right_siblings_are_ordered_ascending_and_left_siblings_descending` |
| C16 | A 6 000-element right-leaning chain materializes without hitting the recursion limit | `tests/unit/test_fugue.py::test_a_deep_chain_does_not_exhaust_the_recursion_limit` |
| C17 | On causally serialized traces the text is identical to pycrdt (Yjs) | `tests/integration/test_pycrdt_differential.py::test_a_serialized_trace_produces_the_same_text_as_pycrdt`, `tests/integration/test_pycrdt_differential.py::test_a_serialized_trace_relayed_through_a_second_replica_still_matches` |
| C18 | Concurrent traces are outside the differential scope; only convergence is asserted there | `tests/integration/test_pycrdt_differential.py::test_concurrent_edits_are_documented_as_outside_the_differential_scope` |

## Suggestions and the accept gate

| # | Claim | Evidence |
|---|---|---|
| C19 | A pending suggestion is invisible in the main view and visible in its own | `tests/property/test_suggestions.py::test_a_pending_suggestion_is_invisible_in_the_main_view`, `tests/unit/test_replica.py::test_a_suggestion_is_invisible_until_it_is_accepted` |
| C20 | Rejecting leaves the main view byte-identical to replaying the history without the suggestion | `tests/property/test_suggestions.py::test_a_rejected_suggestion_leaves_the_main_view_byte_identical`, `tests/integration/test_agent_review.py::test_rejecting_leaves_the_document_exactly_as_it_was` |
| C20a | Scopes are parent-closed against the wire, not only against the local insert path: an element anchored inside a suggestion is invisible in every view that does not contain that suggestion | `tests/unit/test_fugue.py::test_a_suggestion_anchored_to_another_suggestion_is_invisible_without_it`, `tests/unit/test_fugue.py::test_a_main_element_anchored_inside_a_suggestion_is_invisible_in_the_main_view`, `tests/unit/test_document.py::test_a_suggestion_cannot_anchor_itself_inside_another_suggestion` |
| C21 | An accepted run is contiguous and follows its anchor, or the nearest element that survived | `tests/property/test_suggestions.py::test_an_accepted_suggestion_lands_after_its_anchor_or_its_surviving_predecessor` |
| C22 | Accept and reject are ordinary last-writer-wins writes: they commute and the newest wins | `tests/property/test_suggestions.py::test_accepting_and_then_rejecting_is_the_last_writer_that_wins` |
| C23 | Two suggestions at one anchor stay separate; accepting one does not pull in the other | `tests/property/test_suggestions.py::test_two_suggestions_at_one_anchor_stay_separate` |
| C24 | A suggested field write lives in its own register, so rejecting restores the previous main value | `tests/unit/test_lww_orset.py::test_a_suggested_write_is_a_separate_register_so_rejecting_restores_the_main_value` |
| C25 | The suggestion digest covers only that suggestion: concurrent human edits do not change it, more agent operations do | `tests/unit/test_document.py::test_the_suggestion_digest_covers_only_that_suggestion` |
| C25a | Removing one element in the main document and inside a suggestion records both removals, in either arrival order | `tests/unit/test_lww_orset.py::test_removing_one_tag_in_two_scopes_keeps_both_removals_in_either_order`, `tests/unit/test_lww_orset.py::test_a_removal_in_one_scope_leaves_the_other_scope_untouched` |

## Signed operations and the log

| # | Claim | Evidence |
|---|---|---|
| C26 | The replica id is derived from the public key, and a spoofed id is detected | `tests/unit/test_signing.py::test_a_replica_id_is_derived_from_the_public_key`, `tests/unit/test_signing.py::test_a_spoofed_replica_id_is_detected_even_with_a_valid_signature` |
| C27 | A tampered payload fails verification and never reaches the state | `tests/unit/test_signing.py::test_a_tampered_payload_fails_verification`, `tests/unit/test_replica.py::test_a_forged_signature_never_reaches_the_state` |
| C28 | The signature covers the document id and the sequence number | `tests/unit/test_signing.py::test_signatures_cover_the_document_id_so_operations_do_not_move_between_documents`, `tests/unit/test_signing.py::test_signatures_cover_the_sequence_number_so_operations_cannot_be_renumbered` |
| C29 | A spent `(author, seq)` pair cannot be reused with different content | `tests/unit/test_oplog.py::test_a_re_sent_operation_is_a_duplicate_and_a_different_one_is_a_replay`, `tests/unit/test_replica.py::test_a_reused_sequence_number_with_different_content_is_a_replay`, `tests/integration/test_hub.py::test_a_replay_of_a_spent_sequence_number_is_refused` |
| C29a | An operation dated more than 60 s ahead of local wall time is refused by a peer and by the server, and a received timestamp advances the receiver's clock past it | `tests/unit/test_replica.py::test_an_operation_dated_far_in_the_future_is_refused_on_receive`, `tests/integration/test_hub.py::test_an_operation_dated_far_in_the_future_never_reaches_the_log`, `tests/unit/test_replica.py::test_receiving_a_remote_timestamp_pulls_the_local_clock_past_it`, `tests/unit/test_replica.py::test_an_operation_just_inside_the_drift_bound_is_still_accepted` |
| C30 | Editing a log row is detected; so is rebuilding the chain around a forged operation, and so is a row that is not an operation at all — every failure is reported, never raised | `tests/unit/test_oplog.py::test_verification_detects_an_edited_row`, `tests/unit/test_oplog.py::test_verification_detects_a_rebuilt_chain_with_a_forged_signature`, `tests/unit/test_oplog.py::test_verification_reports_a_row_that_is_not_an_operation_at_all`, `tests/integration/test_demos_cli.py::test_doc_verify_reports_a_tampered_row_as_json_instead_of_crashing` |
| C31 | Tail truncation needs an anchored head hash to detect | `tests/unit/test_oplog.py::test_truncating_the_tail_needs_an_anchored_head` |
| C32 | The server rebuilds its state from the log and reaches the same hash | `tests/integration/test_hub.py::test_the_hub_rebuilds_its_state_from_the_log_and_audits_decisions` |
| C33 | Private keys are written to 0600 files and round-trip | `tests/unit/test_signing.py::test_a_key_file_round_trips_and_is_owner_readable_only` |

## Roles, the sync server and its limits

| # | Claim | Evidence |
|---|---|---|
| C34 | A suggester may only write operations carrying a suggestion id | `tests/unit/test_auth.py::test_a_suggester_may_only_write_suggestion_scoped_content`, `tests/integration/test_hub.py::test_a_suggester_cannot_write_main_operations_or_accept_flags`, `tests/integration/test_sync_session.py::test_a_suggester_session_is_refused_the_operations_it_may_not_write` |
| C35 | Only an owner may decide a suggestion; an editor may not | `tests/unit/test_auth.py::test_an_editor_writes_the_document_but_does_not_decide_suggestions`, `tests/integration/test_hub.py::test_an_editor_writes_the_document_but_cannot_decide` |
| C36 | A connection may only publish as the replica its token pins | `tests/unit/test_auth.py::test_a_connection_may_only_publish_as_the_replica_its_token_pins` |
| C37 | Unsigned, expired, future-dated and cross-document tokens are refused | `tests/unit/test_auth.py::test_unsigned_tokens_are_refused_because_the_algorithm_is_pinned`, `tests/unit/test_auth.py::test_expired_and_future_tokens_are_refused`, `tests/unit/test_auth.py::test_a_token_for_another_document_or_secret_is_refused`, `tests/unit/test_auth.py::test_tokens_missing_or_misusing_claims_are_refused` |
| C38 | A WebSocket connection without a usable token is closed before it is accepted | `tests/integration/test_websocket_server.py::test_a_connection_without_a_usable_token_is_closed_before_it_is_accepted`, `tests/integration/test_websocket_server.py::test_a_token_for_another_document_cannot_open_this_one` |
| C39 | A session welcomes, catches up by version vector, pushes and relays | `tests/integration/test_websocket_server.py::test_a_websocket_session_syncs_a_document_end_to_end`, `tests/integration/test_websocket_server.py::test_a_second_client_catches_up_and_receives_relayed_edits`, `tests/integration/test_sync_session.py::test_a_session_welcomes_catches_up_pushes_and_relays` |
| C40 | A replica that was offline catches up and pushes what it wrote while away | `tests/integration/test_sync_session.py::test_a_partitioned_replica_catches_up_when_it_reconnects` |
| C41 | Catch-up returns exactly what a version vector is missing, and is bounded | `tests/integration/test_hub.py::test_catch_up_sends_exactly_what_a_version_vector_is_missing`, `tests/integration/test_hub.py::test_catch_up_is_bounded`, `tests/unit/test_oplog.py::test_catch_up_returns_exactly_what_a_version_vector_is_missing` |
| C42 | A client that stops reading is disconnected rather than buffered | `tests/integration/test_sync_session.py::test_a_client_that_stops_reading_is_disconnected`, `tests/integration/test_hub.py::test_a_subscriber_that_stops_reading_is_marked_for_disconnection` |
| C43 | Too many frames closes the connection | `tests/integration/test_sync_session.py::test_too_many_frames_closes_the_connection` |
| C44 | Pushes, documents, frames and subscriber counts are all bounded | `tests/integration/test_hub.py::test_a_push_larger_than_the_limit_is_refused_whole`, `tests/integration/test_hub.py::test_a_document_stops_growing_at_its_size_limit`, `tests/integration/test_hub.py::test_a_document_refuses_more_subscribers_than_it_allows`, `tests/integration/test_sync_session.py::test_frames_are_bounded_and_unknown_types_are_refused` |
| C45 | Malformed frames are reported without dropping the session; a bad opening frame closes it | `tests/integration/test_sync_session.py::test_a_malformed_frame_mid_session_is_reported_without_dropping_the_session`, `tests/integration/test_sync_session.py::test_the_first_frame_must_be_a_hello_for_this_document`, `tests/integration/test_sync_session.py::test_a_malformed_opening_frame_closes_the_session` |
| C46 | The document endpoint requires an API key and shows the main view plus pending suggestions | `tests/integration/test_websocket_server.py::test_the_document_endpoint_shows_the_main_view_and_requires_a_key` |
| C47 | Metrics require a key and label only bounded dimensions | `tests/integration/test_websocket_server.py::test_metrics_are_exposed_and_require_a_key`, `tests/integration/test_app.py::test_metrics_require_auth_and_label_route_templates` |

## Agents

| # | Claim | Evidence |
|---|---|---|
| C48 | The document reaches the model as untrusted data, with a bounded anchor table | `tests/unit/test_planner.py::test_the_document_reaches_the_model_as_untrusted_data`, `tests/unit/test_planner.py::test_the_anchor_table_is_bounded`, `tests/unit/test_planner.py::test_the_anchor_table_lists_the_start_word_starts_and_the_end` |
| C49 | The planner request is deterministic, so a run is replayable | `tests/unit/test_planner.py::test_the_request_is_deterministic_so_it_can_be_replayed` |
| C50 | An injected directive cannot produce a step the plan schema does not have | `tests/unit/test_planner.py::test_an_injected_directive_cannot_produce_a_step_that_does_not_exist`, `tests/integration/test_agent_review.py::test_an_injected_directive_cannot_produce_a_step_the_schema_does_not_have` |
| C51 | There is no tool for accepting a suggestion, granting a role or reaching the network | `tests/unit/test_agent_tools.py::test_there_is_no_tool_for_accepting_a_suggestion_or_reaching_the_network`, `tests/unit/test_agent_tools.py::test_the_registry_offers_exactly_the_four_edit_tools` |
| C52 | Tool arguments are validated against closed schemas and bounded limits | `tests/unit/test_agent_tools.py::test_arguments_are_validated_against_closed_schemas`, `tests/unit/test_agent_tools.py::test_tool_arguments_carry_the_same_limits_as_the_operations`, `tests/unit/test_agent_tools.py::test_anchors_must_be_dots_or_the_start_sentinel` |
| C53 | A plan addressing an anchor the agent was not shown is rejected, whole | `tests/unit/test_planner.py::test_a_plan_addressing_an_unknown_anchor_is_rejected`, `tests/integration/test_agent_review.py::test_a_plan_addressing_an_element_deleted_concurrently_is_rejected`, `tests/integration/test_agent_review.py::test_a_plan_whose_later_step_deletes_an_element_an_earlier_step_removed` |
| C54 | Step, token and USD budgets stop a runaway planner and survive restarts | `tests/unit/test_planner.py::test_the_step_budget_stops_a_runaway_planner`, `tests/unit/test_budget.py::test_state_survives_restart_and_limits_cannot_be_raised`, `tests/unit/test_budget.py::test_token_overrun_is_recorded_and_blocks_further_steps` |
| C55 | A proposal is invisible and opens a review bound to its digest and its exact text | `tests/integration/test_agent_review.py::test_a_proposal_is_invisible_and_opens_a_review_bound_to_its_digest` |
| C56 | Publishing is refused until a different human approves that exact suggestion, and the approval is single-use | `tests/integration/test_agent_review.py::test_publishing_is_refused_until_a_human_approves_that_exact_suggestion`, `tests/integration/test_agent_review.py::test_an_agent_cannot_approve_its_own_suggestion` |
| C57 | An agent that keeps editing after review loses its approval | `tests/integration/test_agent_review.py::test_an_agent_that_keeps_editing_after_review_loses_its_approval` |
| C58 | The accept gate refuses a suggestion that changed between review and decision | `tests/integration/test_agent_review.py::test_the_document_gate_refuses_a_suggestion_that_changed_after_review` |
| C59 | A denied suggestion never reaches the document | `tests/integration/test_agent_review.py::test_a_denied_suggestion_never_reaches_the_document` |
| C60 | A suggester cannot push an accept flag even when it signs one correctly | `tests/integration/test_agent_review.py::test_a_suggester_cannot_push_the_accept_flag_even_if_it_forges_the_operation` |
| C61 | Accepting makes the suggestion part of the document, and the audit chain stays valid | `tests/integration/test_agent_review.py::test_accepting_makes_the_suggestion_part_of_the_document` |

## The Anthropic provider

| # | Claim | Evidence |
|---|---|---|
| C62 | Offline code paths never import the SDK | `tests/unit/test_anthropic_provider.py::test_offline_modules_do_not_import_the_sdk` |
| C63 | `stop_reason` is classified before content is read, including refusals and truncation | `tests/unit/test_anthropic_provider.py::test_refusal_is_raised_before_content_is_read`, `tests/unit/test_anthropic_provider.py::test_non_final_stop_reasons_raise` |
| C64 | The server-side refusal fallback is requested only for the models documented for it | `tests/unit/test_anthropic_provider.py::test_refusal_fallback_is_sent_only_to_models_documented_to_accept_it` |
| C65 | Blocks from a declined attempt before the fallback marker are neither executed nor echoed | `tests/unit/test_anthropic_provider.py::test_blocks_of_a_declined_attempt_before_the_fallback_marker_are_dropped` |
| C66 | Cache tokens and billed failures are charged to the budget | `tests/unit/test_anthropic_provider.py::test_cache_tokens_are_charged_as_input_tokens`, `tests/unit/test_anthropic_provider.py::test_billed_responses_that_fail_validation_raise_provider_errors_with_usage` |
| C67 | SDK errors map most-specific-first onto typed, retry-aware exceptions | `tests/unit/test_anthropic_provider.py::test_sdk_errors_map_to_typed_provider_errors` |

## Snapshots, the CLI and the demo

| # | Claim | Evidence |
|---|---|---|
| C68 | Snapshots are written atomically and verified against their recorded hash on load | `tests/unit/test_snapshot_netsim.py::test_a_snapshot_round_trips_and_carries_the_version_vector`, `tests/unit/test_snapshot_netsim.py::test_a_corrupted_snapshot_is_refused`, `tests/unit/test_snapshot_netsim.py::test_writing_a_snapshot_never_leaves_a_partial_file` |
| C69 | A restored snapshot is a working replica | `tests/unit/test_snapshot_netsim.py::test_a_snapshot_restores_into_a_working_replica` |
| C70 | The offline demo converges, accepts one suggestion and rejects the other with no trace | `tests/integration/test_demos_cli.py::test_the_demo_converges_accepts_one_suggestion_and_rejects_the_other` |
| C71 | Running the demo twice starts from a clean state | `tests/integration/test_demos_cli.py::test_running_the_demo_twice_starts_from_a_clean_state` |
| C72 | The CLI reads and verifies a local log, writes snapshots and lists reviews | `tests/integration/test_demos_cli.py::test_doc_show_and_verify_read_the_local_log`, `tests/integration/test_demos_cli.py::test_doc_snapshot_writes_a_verified_file`, `tests/integration/test_demos_cli.py::test_the_review_and_accept_commands_drive_the_two_gates` |
| C73 | `audit-verify` detects a tampered review log | `tests/integration/test_demos_cli.py::test_audit_verify_detects_a_tampered_log` |
| C74 | `serve` refuses to start without a signing secret and configures uvicorn safely | `tests/integration/test_demos_cli.py::test_serve_refuses_to_start_without_a_signing_secret`, `tests/integration/test_demos_cli.py::test_serve_configures_uvicorn_safely` |
| C75 | `keygen` will not overwrite an existing key, and `token` needs a secret | `tests/integration/test_demos_cli.py::test_keygen_writes_a_key_once_and_reports_its_replica_id`, `tests/integration/test_demos_cli.py::test_token_needs_a_secret_and_mints_a_usable_grant` |

## Runtime hygiene inherited from the baseline

| # | Claim | Evidence |
|---|---|---|
| C76 | API keys are stored as digests and compared in constant time against every configured key | `tests/unit/test_security.py::test_every_configured_digest_is_compared_without_early_exit`, `tests/integration/test_app.py::test_protected_endpoint_requires_a_valid_api_key` |
| C77 | Errors never leak exception text, tracebacks or submitted values | `tests/integration/test_app.py::test_unhandled_exceptions_become_generic_500s`, `tests/integration/test_app.py::test_validation_errors_never_echo_submitted_values` |
| C78 | Logs, tracebacks and audit records pass through PII and secret redaction | `tests/unit/test_logging.py::test_exception_tracebacks_are_redacted`, `tests/unit/test_audit.py::test_entries_are_redacted_before_hashing` |
| C79 | Rate limits, body limits, security headers and strict CORS are applied | `tests/integration/test_app.py::test_rate_limit_rejects_bursts_and_exempts_probes`, `tests/integration/test_app.py::test_body_size_limit_applies_to_declared_and_streamed_bodies`, `tests/integration/test_app.py::test_cors_allows_only_configured_origins` |
| C80 | Readiness reports check names without error details | `tests/integration/test_app.py::test_readyz_reports_check_names_without_error_details`, `tests/integration/test_websocket_server.py::test_readiness_reports_the_operation_log` |
| C81 | The audit log is a hash chain: any single-byte change is detected | `tests/unit/test_audit.py::test_any_single_byte_change_is_detected`, `tests/unit/test_audit.py::test_tail_truncation_needs_an_anchored_head` |
| C82 | Nothing in the source tree uses pickle-family serialization | `tests/unit/test_no_pickle.py::test_source_tree_has_no_pickle_family_serialization` |
| C83 | `.env.example` documents every setting | `tests/unit/test_config.py::test_env_example_documents_every_setting` |

## Numbers in the README

| # | Claim | Evidence |
|---|---|---|
| C84 | Local insert, remote apply, merge-only and materialize latencies | `bench/bench_crdt.py`, enforced in `.github/workflows/ci.yml#bench` |
| C85 | State memory at 100 000 elements, measured with `tracemalloc` | `bench/bench_crdt.py` |
| C86 | Delta bytes against a full-state transfer at 1, 10 and 100 operations behind | `bench/bench_crdt.py` |
| C87 | 5 replicas × 400 operations in five different random orders converge to one hash and one view | `bench/bench_crdt.py`, `tests/property/test_convergence.py::test_five_replicas_converge_over_ten_thousand_operations` |
| C88 | Baseline component latencies (redaction, rate limiter, request hash, audit) | `bench/bench_baseline.py`, enforced in `.github/workflows/ci.yml#bench` |
| C89 | Coverage stays at or above 85% | `.github/workflows/ci.yml#test` |
| C90 | The convergence properties also run nightly with 10× the examples | `.github/workflows/ci.yml#nightly-properties` |

## CI

| # | Claim | Evidence |
|---|---|---|
| C91 | ruff, mypy --strict, bandit, semgrep, pip-audit and gitleaks run on every change | `.github/workflows/ci.yml#lint`, `.github/workflows/ci.yml#typecheck`, `.github/workflows/ci.yml#bandit`, `.github/workflows/ci.yml#semgrep`, `.github/workflows/ci.yml#pip-audit`, `.github/workflows/ci.yml#gitleaks` |
| C92 | The container is built, smoke-tested as non-root, scanned by trivy and inventoried by syft | `.github/workflows/ci.yml#docker` |
| C93 | The baseline structure check runs in CI and fails on a dangling evidence reference | `.github/workflows/ci.yml#baseline`, `tests/integration/test_check_baseline.py::test_dangling_claim_reference_fails` |
