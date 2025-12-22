# run_review

`run_review` is an offline analyzer that turns pipeline artifacts into a structured JSON summary and a human-friendly HTML report. It works even as pipeline schemas evolve by joining data on step paths instead of relying on indices.

## Usage

```bash
python -m run_review.cli --generation-id <id> --logs-dir <artifact_dir>
python -m run_review.cli --oplog /path/to/oplog.log --transcript /path/to/transcript.json [--generation-id <id>]
python -m run_review.cli --best-effort --generation-id <id> --logs-dir <artifact_dir>
python -m run_review.cli --compare <runA> <runB> --logs-dir <artifact_dir>
```

The tool writes `<generation_id>_run_report.json` and `<generation_id>_run_report.html` (or `<runA>_vs_<runB>_run_compare.*` in compare mode) to the current directory by default. Use `--output-dir` to change the destination.

## Behavior

- **Discovery:** When only a generation ID is supplied, the tool looks for `<id>_oplog.log` and `<id>_transcript.json` inside `--logs-dir`.
- **Best effort:** By default the tool requires both oplog + transcript; when `--best-effort` is set, missing artifacts produce warnings instead of hard failures.
- **Parsing:** The oplog parser extracts run boundaries, config defaults, seed selection, context injection, step start/end markers, image generation, upscaling, manifest appends, uploads, and file writes. Unrecognized lines are retained under `unknown_events`.
- **Transcript:** The transcript parser tolerates older schemas and surfaces optional fields like `context`, `title_generation`, and `concept_filter_log` when present.
- **Joining:** Steps are merged by their `pipeline/...` paths. Timing data comes from the oplog when available, while prompts/responses and size counters come from the transcript.
- **Issues:** The JSON report includes structured `issues[]` for missing artifacts, unmatched steps, missing start/end markers, empty responses, large contexts, config defaults, concept filter no-ops, and other anomalies.
- **Side effects:** Image generation, upscaling, manifest, and upload events are surfaced in both HTML and JSON outputs.
- **Comparison:** `--compare` produces a schema-tolerant diff that highlights added/removed steps, shifts in metadata presence (e.g., `concept_filter_log`), injector messaging changes, and post-processing log format changes (e.g., upscaling).

## Files produced

- `*_run_report.json`: machine-readable run summary
- `*_run_report.html`: human-friendly report for manual review
- `*_run_compare.json` / `*_run_compare.html`: diff artifacts when `--compare` is used
