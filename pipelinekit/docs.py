"""`pipelinekit` invariants and boundaries.

This module exists to make repository-wide refactors and boundary tests explicit.

Generic invariants:

1) `pipelinekit` must not import `image_project.*`.
2) `pipelinekit` provides reusable engine primitives (step/block execution + recording)
   and a stage authoring kit (StageRef/StageRegistry/ConfigNamespace).
3) `pipelinekit` does not define project conventions like:
   - which steps inside a stage are "primary"
   - how stage instance selectors (include/exclude/overrides) should resolve
   - what it means to "capture" a final output (and which stage is the default capture)

Project code should inject conventions via explicit policy objects implemented outside
this package.
"""

