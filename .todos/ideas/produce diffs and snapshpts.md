# Produce diffs and “repo snapshots” (notes for future tooling)

This file captures the manual steps used during the `StageSpec -> stage Block` refactor so we can later automate them in a small tool.

## 1) Capture a diff to a file (what I did)

From the repo root (`i:\code\image_project`):

```powershell
git diff --no-color > .todos\work\rafactor_implementation.diff
```

Notes:
- `--no-color` avoids ANSI escape codes in the saved diff file.
- This captures *working tree* changes to tracked files. It does not include untracked files (by design).
- If we ever need staged changes instead: `git diff --cached --no-color`.
- If we ever need “everything vs HEAD” (staged + unstaged): `git diff --no-color HEAD`.

## 2) Capture a local repo snapshot into a zip (excluding gitignored)

Goal: produce a zip that contains the current repo contents you care about for debugging/repro, without accidentally scooping up ignored artifacts.

High-level approach (don’t overthink it):
1. Ask git for the file list: tracked + untracked, excluding ignored.
2. Zip exactly that list, preserving repo-relative paths.
3. Record minimal provenance alongside the zip (timestamp, `HEAD`, dirty flag, and optionally the diff file path).

Git file list command:

```bash
git ls-files -z --cached --others --exclude-standard
```

Implementation sketch (cross-platform friendly):
- Use a small Python helper:
  - `subprocess.check_output(["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"])`
  - split on `\0` to get repo-relative paths
  - write to `zipfile.ZipFile(...)` with `arcname=<repo-relative-path>` so the zip unpacks cleanly
  - sort paths for determinism
  - fail fast if `git` fails, if not in a git repo, or if any listed path can’t be read

Practical footguns to avoid (tool should guard these):
- Don’t include the output zip in itself (write the zip under an ignored dir like `_artifacts/` or pass an explicit exclude for the output filename).
- Don’t silently skip unreadable files; error with a clear path (repo invariant: no silent fallbacks).

Optional nice-to-haves (not required for v1):
- Add a small JSON sidecar in the zip root (e.g. `snapshot_meta.json`) containing:
  - `created_at`, `git_head`, `git_branch` (if available), `git_status_porcelain`
  - tool version + args
  - path to the diff file if also produced
