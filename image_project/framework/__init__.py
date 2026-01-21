"""Project-specific framework utilities.

This package contains structural helpers that are generic *within* this repo
(artifact writing/indexing, run context contracts, plan/prompt pipeline helpers),
but intentionally excludes prompt template text and stage/plan implementations.

Common entrypoints:

- `image_project.framework.artifacts`: manifest/transcript writing + artifacts indexing
- `image_project.framework.prompt_pipeline`: prompt pipeline authoring + compilation glue

For reusable, project-agnostic pipeline primitives, use `pipelinekit`.
"""
