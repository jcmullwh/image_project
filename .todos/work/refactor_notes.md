# Notes

### Mid-phase iic notes

framework/artifacts/index.py looks insanely brittle. It lists a ton of keys etc that are specific to the current implementation. It seems like it should be in impl/current instead, or at least be more generic. Maybe it depends on how stable it actually is. Is it because there's a fuzziness still on framework vs impl vs app?


framework/artifacts/manifest.py is mostly generating titles. And does it a bit awkwardly. Low priority but it feels like it could be improved.

config.py is still a giant file of crap. It has a bunch of hard encoded stuff. Like prompt_schema that has every single stage config key? Why is that there? Have we just not refactored it to the stage level config? There are things like PromptBlackboxRefineConfig. Has it just not been moved? It also has upscale, context injection, etc. but I think that's just because we haven't specifically refactored those into stages which I think we should. Then at the end we end up with RunConfig that just lists everything. Aren't we supposed to be moving away from that? Where everything config related to a stage is owned and lives in the stage module? 

Partially related to previous: looking at context.py; is there a conflict in architecture with making it a stage? Prompts and config live in stages but context.py is in framework. Does that mean it has to live in config?

inputs.py: Maybe it's that framework isn't well organized? Or that stages should be a submodule of framework? This also feels like the core workings of a stage. Although it has two general chunks: resolve_prompt_inputs which is very generic framework-y and then ConceptFilterOutcome and associated which is the part that is very stage-y.

media.py is the core workings of a stage. Although maybe it could hypothetically be used by multiple stages. This feels like another one where the architecture is fuzzy. And the main thing "where things live" being not extremely obvious will continue to result in things being spread out all over the place.

framework/prompt_pipeline/__init__.py seems like it's just named completely incorrectly.


framework/scoring.py is also the core workings of a stage. Although maybe it could definitely be used by multiple stages. Again fuzzy architecture.

pipelinekit/stage_registry.py seems correct for the kernel.

pipelinekit/stage_types.py seems correct for the kernel.

framework/artifacts/transcript.py seems correct for framework BUT seems written in a very non-generic way.


---

Additional:

foundation is things that are very generic and could be used by any project. 

framework is things that are specific to this project but are generic within the project. 

impl/current is things that are specific to the current implementation of the project. 

app is the actual application entry points.

It seems like stages and prompts should be under framework maybe? Or maybe framework should just be very small and that's okay. It seems like impl should basically just be plans (and where the majority of experimentation lives).

We discussed "promoting" things that stabilize. I wonder if we should just have plugins for everything that is not foundation or framework. So stages would be plugins. Plans would be plugins. Then we would have a default set of stages and plans that are shipped with the framework. Then when something stabilizes we can promote it into the framework proper. Two counterpoints: may be non-obvious where things live. Also may need to go to a bunch of different places when reviewing a plan.
