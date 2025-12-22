from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Literal, Protocol, TypeAlias

from message_handling import MessageHandler

from run_config import RunConfig

MergeMode: TypeAlias = Literal["all_messages", "last_response", "none"]
ALLOWED_MERGE_MODES: tuple[str, ...] = ("all_messages", "last_response", "none")


def utc_now_iso8601() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


@dataclass(frozen=True)
class ChatStep:
    name: str | None
    prompt: str | Callable[["RunContext"], str]
    temperature: float
    merge: MergeMode = "all_messages"
    allow_empty_prompt: bool = False
    allow_empty_response: bool = False
    capture_key: str | None = None
    params: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.name is not None:
            if not isinstance(self.name, str):
                raise TypeError(
                    f"Step name must be a string or None (type={type(self.name).__name__})"
                )
            name = self.name.strip()
            if not name:
                raise ValueError("Step name cannot be empty")
            object.__setattr__(self, "name", name)
        if self.capture_key is not None:
            if not isinstance(self.capture_key, str):
                raise TypeError(
                    f"Step capture_key must be a string or None (type={type(self.capture_key).__name__})"
                )
            capture_key = self.capture_key.strip()
            if not capture_key:
                raise ValueError("Step capture_key cannot be empty")
            object.__setattr__(self, "capture_key", capture_key)
        if self.merge not in ALLOWED_MERGE_MODES:
            raise ValueError(f"Invalid step merge mode: {self.merge}")
        if "temperature" in self.params:
            label = self.name or "<unnamed>"
            raise ValueError(
                f"Step {label} sets params['temperature']; use the ChatStep.temperature field instead"
            )

    def render_prompt(self, ctx: "RunContext", *, step_name: str | None = None) -> str:
        label = step_name or self.name or "<unnamed>"
        if callable(self.prompt):
            text = self.prompt(ctx)
        else:
            text = self.prompt

        if text is None:
            raise ValueError(f"Step {label} produced None prompt")
        if not isinstance(text, str):
            raise TypeError(
                f"Step {label} produced non-string prompt (type={type(text).__name__})"
            )
        if not text.strip() and not self.allow_empty_prompt:
            raise ValueError(f"Step {label} produced empty prompt")
        return text


@dataclass(frozen=True)
class Block:
    name: str | None = None
    nodes: list["Node"] = field(default_factory=list)
    merge: MergeMode = "all_messages"
    capture_key: str | None = None

    def __post_init__(self) -> None:
        if self.name is not None:
            if not isinstance(self.name, str):
                raise TypeError(
                    f"Block name must be a string or None (type={type(self.name).__name__})"
                )
            name = self.name.strip()
            if not name:
                raise ValueError("Block name cannot be empty")
            object.__setattr__(self, "name", name)
        if self.merge not in ALLOWED_MERGE_MODES:
            raise ValueError(f"Invalid block merge mode: {self.merge}")
        if self.capture_key is not None:
            if not isinstance(self.capture_key, str):
                raise TypeError(
                    f"Block capture_key must be a string or None (type={type(self.capture_key).__name__})"
                )
            capture_key = self.capture_key.strip()
            if not capture_key:
                raise ValueError("Block capture_key cannot be empty")
            object.__setattr__(self, "capture_key", capture_key)


Node: TypeAlias = ChatStep | Block


@dataclass
class RunContext:
    generation_id: str
    cfg: RunConfig
    logger: logging.Logger
    rng: random.Random
    seed: int
    created_at: str

    messages: MessageHandler
    user_role: str = "user"
    agent_role: str = "assistant"

    selected_concepts: list[str] = field(default_factory=list)
    outputs: dict[str, Any] = field(default_factory=dict)
    blackbox_scoring: dict[str, Any] | None = None
    steps: list[dict[str, Any]] = field(default_factory=list)

    image_path: str | None = None
    error: dict[str, Any] | None = None


class StepRecorder(Protocol):
    def on_step_start(
        self,
        ctx: RunContext,
        path: str,
        **metrics: Any,
    ) -> None:
        ...

    def on_step_end(self, ctx: RunContext, record: dict[str, Any]) -> None:
        ...

    def on_step_error(self, ctx: RunContext, path: str, step_name: str, exc: Exception) -> None:
        ...


class DefaultStepRecorder:
    def on_step_start(
        self,
        ctx: RunContext,
        path: str,
        **metrics: Any,
    ) -> None:
        ctx.logger.info(
            "Step: %s (context_chars=%d, prompt_chars=%d, input_chars=%d)",
            path,
            metrics.get("context_chars", 0),
            metrics.get("prompt_chars", 0),
            metrics.get("input_chars", 0),
        )

    def on_step_end(self, ctx: RunContext, record: dict[str, Any]) -> None:
        ctx.steps.append(record)
        path = record.get("path", "<unknown>")
        input_chars = record.get("input_chars", 0)
        response_chars = record.get("response_chars", len(str(record.get("response", ""))))
        ctx.logger.info(
            "Received response for %s (input_chars=%d, chars=%d)",
            path,
            input_chars,
            response_chars,
        )

    def on_step_error(self, ctx: RunContext, path: str, step_name: str, exc: Exception) -> None:
        ctx.logger.error("Step failed: %s (%s)", path, exc)


class NullStepRecorder:
    def on_step_start(
        self,
        ctx: RunContext,
        path: str,
        **metrics: Any,
    ) -> None:
        return

    def on_step_end(self, ctx: RunContext, record: dict[str, Any]) -> None:
        return

    def on_step_error(self, ctx: RunContext, path: str, step_name: str, exc: Exception) -> None:
        return


class ChatRunner:
    def __init__(self, *, ai_text: Any, recorder: StepRecorder | None = None):
        self._ai_text = ai_text
        self._recorder = recorder or DefaultStepRecorder()
        self._validate_recorder(self._recorder)

    def run(self, ctx: RunContext, node: Node) -> None:
        root_name = self._effective_root_name(node)
        root_path = [root_name]

        produced_messages, last_assistant = self._execute_node(
            ctx, node, parent_messages=ctx.messages, path_segments=root_path
        )
        merge_mode: MergeMode = node.merge
        try:
            self._apply_merge(
                ctx,
                ctx.messages,
                merge_mode=merge_mode,
                produced_messages=produced_messages,
                last_assistant=last_assistant,
            )
        except Exception as exc:
            self._attach_pipeline_error(
                exc,
                pipeline_path="/".join(root_path),
                pipeline_node_type="block" if isinstance(node, Block) else "step",
                pipeline_node_name=root_name,
            )
            raise

    def run_steps(self, ctx: RunContext, steps: list[ChatStep]) -> None:
        root = Block(name="pipeline", merge="all_messages", nodes=list(steps))
        self.run(ctx, root)

    def run_step(self, ctx: RunContext, step: ChatStep) -> str:
        step_name = step.name or "step_01"
        produced, last_assistant = self._execute_node(
            ctx, step, parent_messages=ctx.messages, path_segments=[step_name]
        )
        self._apply_merge(
            ctx,
            ctx.messages,
            merge_mode=step.merge,
            produced_messages=produced,
            last_assistant=last_assistant,
        )
        if last_assistant is None:
            raise ValueError("Step execution produced no assistant output")
        return last_assistant

    def _effective_root_name(self, node: Node) -> str:
        if isinstance(node, Block):
            return node.name or "pipeline"
        return node.name or "step_01"

    def _effective_child_name(self, node: Node, *, index: int) -> str:
        if isinstance(node, ChatStep):
            return node.name or f"step_{index + 1:02d}"
        return node.name or f"block_{index + 1:02d}"

    def _validate_recorder(self, recorder: StepRecorder) -> None:
        required = ("on_step_start", "on_step_end", "on_step_error")
        for name in required:
            method = getattr(recorder, name, None)
            if method is None or not callable(method):
                raise TypeError(f"Step recorder missing required method: {name}")

    def _attach_pipeline_error(
        self,
        exc: Exception,
        *,
        pipeline_path: str,
        pipeline_node_type: str,
        pipeline_node_name: str,
    ) -> None:
        if not hasattr(exc, "pipeline_path"):
            try:
                setattr(exc, "pipeline_path", pipeline_path)
            except Exception:
                pass
        if not hasattr(exc, "pipeline_node_type"):
            try:
                setattr(exc, "pipeline_node_type", pipeline_node_type)
            except Exception:
                pass
        if not hasattr(exc, "pipeline_node_name"):
            try:
                setattr(exc, "pipeline_node_name", pipeline_node_name)
            except Exception:
                pass
        if not hasattr(exc, "pipeline_step"):
            try:
                setattr(exc, "pipeline_step", pipeline_node_name)
            except Exception:
                pass

    def _validate_sibling_names(
        self, *, block: Block, block_path: list[str], effective_children: list[str]
    ) -> None:
        seen: set[str] = set()
        duplicates: set[str] = set()
        for name in effective_children:
            if name in seen:
                duplicates.add(name)
            seen.add(name)
        if duplicates:
            raise ValueError(
                f"Duplicate node name(s) in block {'/'.join(block_path)}: {', '.join(sorted(duplicates))}"
            )

    def _execute_node(
        self,
        ctx: RunContext,
        node: Node,
        *,
        parent_messages: MessageHandler,
        path_segments: list[str],
    ) -> tuple[list[dict[str, Any]], str | None]:
        if isinstance(node, ChatStep):
            return self._execute_step(
                ctx,
                node,
                parent_messages=parent_messages,
                path_segments=path_segments,
            )
        return self._execute_block(
            ctx,
            node,
            parent_messages=parent_messages,
            path_segments=path_segments,
        )

    def _execute_step(
        self,
        ctx: RunContext,
        step: ChatStep,
        *,
        parent_messages: MessageHandler,
        path_segments: list[str],
    ) -> tuple[list[dict[str, Any]], str]:
        pipeline_path = "/".join(path_segments)
        step_name = path_segments[-1] if path_segments else (step.name or "<unnamed>")
        try:
            working = parent_messages.copy()
            prompt_text = step.render_prompt(ctx, step_name=step_name)

            prompt_chars = len(prompt_text)
            context_messages = len(working.messages)
            context_chars = sum(
                len(message.get("content", ""))
                for message in working.messages
                if isinstance(message, dict)
            )
            input_messages = context_messages + 1
            input_chars = context_chars + prompt_chars

            self._recorder.on_step_start(
                ctx,
                pipeline_path,
                context_chars=context_chars,
                prompt_chars=prompt_chars,
                input_chars=input_chars,
                context_messages=context_messages,
                input_messages=input_messages,
            )

            call_params: dict[str, Any] = dict(step.params)
            call_params["temperature"] = step.temperature
            record_params = dict(call_params)
            model_name = getattr(self._ai_text, "model", None) or getattr(
                self._ai_text, "model_name", None
            )
            if model_name:
                if "model" not in record_params:
                    record_params["model"] = model_name
                elif record_params.get("model") != model_name:
                    record_params["base_model"] = model_name

            messages_for_call = [
                *working.messages,
                {"role": ctx.user_role, "content": prompt_text},
            ]

            try:
                response = self._ai_text.text_chat(messages_for_call, **call_params)
            except Exception:
                ctx.logger.exception("AI text chat failed for step %s", pipeline_path)
                raise

            if response is None:
                raise ValueError(f"Step {step_name} produced None response")
            if not isinstance(response, str):
                raise TypeError(
                    f"Step {step_name} produced non-string response (type={type(response).__name__})"
                )
            if not response.strip() and not step.allow_empty_response:
                raise ValueError(f"Step {step_name} produced empty response")

            response_text = response

            working.continue_messages(ctx.user_role, prompt_text)
            working.continue_messages(ctx.agent_role, response_text)

            produced_messages = working.messages[len(parent_messages.messages) :]

            record = {
                "name": step_name,
                "path": pipeline_path,
                "prompt": prompt_text,
                "response": response_text,
                "params": record_params,
                "prompt_chars": prompt_chars,
                "response_chars": len(response_text),
                "context_chars": context_chars,
                "input_chars": input_chars,
                "context_messages": context_messages,
                "input_messages": input_messages,
                "created_at": utc_now_iso8601(),
            }

            self._recorder.on_step_end(ctx, record)

            if step.capture_key:
                ctx.outputs[step.capture_key] = response_text

            return produced_messages, response_text
        except Exception as exc:
            try:
                self._recorder.on_step_error(ctx, pipeline_path, step_name, exc)
            except Exception:
                ctx.logger.exception("Step recorder failed during error handling for %s", pipeline_path)
            self._attach_pipeline_error(
                exc,
                pipeline_path=pipeline_path,
                pipeline_node_type="step",
                pipeline_node_name=step_name,
            )
            raise

    def _execute_block(
        self,
        ctx: RunContext,
        block: Block,
        *,
        parent_messages: MessageHandler,
        path_segments: list[str],
    ) -> tuple[list[dict[str, Any]], str | None]:
        block_name = path_segments[-1] if path_segments else (block.name or "<unnamed>")
        pipeline_path = "/".join(path_segments)
        try:
            working = parent_messages.copy()

            effective_children: list[str] = [
                self._effective_child_name(child, index=i) for i, child in enumerate(block.nodes)
            ]
            self._validate_sibling_names(
                block=block, block_path=path_segments, effective_children=effective_children
            )

            last_assistant: str | None = None

            for child, child_name in zip(block.nodes, effective_children, strict=False):
                child_path = [*path_segments, child_name]
                child_produced, child_last_assistant = self._execute_node(
                    ctx,
                    child,
                    parent_messages=working,
                    path_segments=child_path,
                )

                merge_mode: MergeMode = child.merge
                try:
                    self._apply_merge(
                        ctx,
                        working,
                        merge_mode=merge_mode,
                        produced_messages=child_produced,
                        last_assistant=child_last_assistant,
                    )
                except Exception as exc:
                    self._attach_pipeline_error(
                        exc,
                        pipeline_path="/".join(child_path),
                        pipeline_node_type="block" if isinstance(child, Block) else "step",
                        pipeline_node_name=child_name,
                    )
                    raise

                if merge_mode != "none" and child_last_assistant is not None:
                    last_assistant = child_last_assistant

            produced_messages = working.messages[len(parent_messages.messages) :]
            last_assistant = self._last_assistant_from_delta(ctx, produced_messages) or last_assistant

            if block.capture_key:
                if last_assistant is None:
                    raise ValueError(
                        f"Block {pipeline_path} capture_key={block.capture_key} requested but no assistant output exists"
                    )
                ctx.outputs[block.capture_key] = last_assistant

            return produced_messages, last_assistant
        except Exception as exc:
            self._attach_pipeline_error(
                exc,
                pipeline_path=pipeline_path,
                pipeline_node_type="block",
                pipeline_node_name=block_name,
            )
            raise

    def _last_assistant_from_delta(
        self, ctx: RunContext, produced_messages: list[dict[str, Any]]
    ) -> str | None:
        for message in reversed(produced_messages):
            if message.get("role") == ctx.agent_role:
                return message.get("content")  # type: ignore[return-value]
        return None

    def _apply_merge(
        self,
        ctx: RunContext,
        parent_working: MessageHandler,
        *,
        merge_mode: MergeMode,
        produced_messages: list[dict[str, Any]],
        last_assistant: str | None,
    ) -> None:
        if merge_mode == "all_messages":
            parent_working.messages.extend(produced_messages)
            return
        if merge_mode == "none":
            return
        if merge_mode == "last_response":
            if last_assistant is None or not str(last_assistant).strip():
                raise ValueError("last_response requested but no assistant output exists")
            parent_working.continue_messages(ctx.agent_role, last_assistant)
            return

        raise ValueError(f"Invalid merge mode: {merge_mode}")
