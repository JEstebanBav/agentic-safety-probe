RERUN INPUTS v0.3

1. Input file: dataset_v03_rerun.jsonl (96 pairs, 192 requests, 384 variants).
From the existing repository, use your model/run options with:
  --dataset custom --dataset-path /path/to/dataset_v03_rerun.jsonl

2. Send only prompt, system_prompt and tool_definitions to the model, using the existing format_entry path. IDs, variant, category and is_harmful are control metadata, not prompt content. Never serialize the full record as the user message.

3. If using --toolbox, role_plus_tools, history, trajectory or any default/global tool catalogue, replace BOTH repository catalogue files before starting a fresh process:
  catalogos_neutrales/tool_definitions.json -> <repository>/tool_definitions.json
  catalogos_neutrales/data/tool_definitions_agentharm.json -> <repository>/data/tool_definitions_agentharm.json
Keep backups of the previous catalogues outside these active paths. The second file is intentionally empty: the full neutral union is in the first. No original repository file was changed while preparing this export.
Use the per-record schemas for the paired chat/agent run. Global variants have their own rendering rules and should be recorded separately. The neutral union preserves extra tool capabilities, with harmful-specific encryption variants consolidated into encrypt_files. No claim of backend compatibility or tool execution is made; these are model-facing specifications.

4. is_harmful is the ORIGINAL intended experimental arm, copied only because the current loader requires a boolean. It is NOT an adjudicated human intent label. category is also legacy metadata. No missing final intent or explicitness labels were filled. subtlety is omitted, so the legacy loader returns its internal default unknown; that value is not sent to the model. Do not interpret run-time arm groupings or that default as new annotation results. There are no review notes, ambiguity suggestions, source prompt copies, Spanish support or annotation-status fields in the input file.

5. Preserve IDs when saving generations and activations. Keep members of theme_groups_67.csv in the same train/test partition; the existing runner must explicitly use these groups for this constraint to hold. Keep this run separate from older outputs. Save the exact tokenizer/model revision, rendered inputs, generation configuration, input file hash and active catalogue hash.

6. Checks completed: actual repository loader loads all 384 rows; both-arm schemas and systems are identical within each pair; no listed harmful-specific tool cues or annotation status text remain in the exported inputs/catalogues; formatting boundary checked for native tools and text fallback using template harnesses. Actual model-tokenizer rendering and GPU execution were not performed here. Source draft and original catalogues are unchanged. Final human annotation remains separate from this generation export.
