"""Execution engine for Block/Step trees.

This module is intentionally app-agnostic and must not import `image_project.*`.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Literal, Protocol, TypeAlias

from .messages import MessageHandler

MergeMode: TypeAlias = Literal["all_messages", "last_response", "none"]
ALLOWED_MERGE_MODES: tuple[str, ...] = ("all_messages", "last_response", "none")


def utc_now_iso8601() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


class FlowContext(Protocol):
    logger: logging.Logger
    messages: MessageHandler
    user_role: str
    agent_role: str
    outputs: dict[str, Any]
    steps: list[dict[str, Any]]


@dataclass(frozen=True)
class ChatStep:
    name: str | None
    prompt: str | Callable[[FlowContext], str]
    temperature: float
    merge: MergeMode = "all_messages"
    allow_empty_prompt: bool = False
    allow_empty_response: bool = False
    capture_key: str | None = None
    params: dict[str, Any] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

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
        if not isinstance(self.meta, dict):
            raise TypeError(f"Step meta must be a dict (type={type(self.meta).__name__})")

    def render_prompt(self, ctx: FlowContext, *, step_name: str | None = None) -> str:
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
    meta: dict[str, Any] = field(default_factory=dict)

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
        if not isinstance(self.meta, dict):
            raise TypeError(f"Block meta must be a dict (type={type(self.meta).__name__})")


@dataclass(frozen=True)
class ActionStep:
    """Pure-Python glue execution node (non-LLM)."""

    name: str | None
    fn: Callable[[FlowContext], Any]
    merge: MergeMode = "none"
    capture_key: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.name is not None:
            if not isinstance(self.name, str):
                raise TypeError(
                    f"Action name must be a string or None (type={type(self.name).__name__})"
                )
            name = self.name.strip()
            if not name:
                raise ValueError("Action name cannot be empty")
            object.__setattr__(self, "name", name)

        if not callable(self.fn):
            raise TypeError(f"Action fn must be callable (type={type(self.fn).__name__})")

        if self.merge not in ALLOWED_MERGE_MODES:
            raise ValueError(f"Invalid action merge mode: {self.merge}")
        if self.merge == "last_response":
            raise ValueError("ActionStep does not support merge='last_response'")

        if self.capture_key is not None:
            if not isinstance(self.capture_key, str):
                raise TypeError(
                    f"Action capture_key must be a string or None (type={type(self.capture_key).__name__})"
                )
            capture_key = self.capture_key.strip()
            if not capture_key:
                raise ValueError("Action capture_key cannot be empty")
            object.__setattr__(self, "capture_key", capture_key)

        if not isinstance(self.meta, dict):
            raise TypeError(f"Action meta must be a dict (type={type(self.meta).__name__})")


Node: TypeAlias = ChatStep | Block | ActionStep


class StepRecorder(Protocol):
    def on_step_start(
        self,
        ctx: FlowContext,
        path: str,
        **metrics: Any,
    ) -> None:
        ...

    def on_step_end(self, ctx: FlowContext, record: dict[str, Any]) -> None:
        ...

    def on_step_error(
        self, ctx: FlowContext, path: str, step_name: str, exc: Exception
    ) -> None:
        ...


class DefaultStepRecorder:
    def on_step_start(
        self,
        ctx: FlowContext,
        path: str,
        **metrics: Any,
    ) -> None:
        tokens: list[str] = []
        node_type = metrics.get("node_type")
        if isinstance(node_type, str) and node_type.strip():
            tokens.append(f"type={node_type.strip()}")

        stage_id = metrics.get("stage_id")
        if isinstance(stage_id, str) and stage_id.strip():
            tokens.append(f"stage_id={stage_id.strip()}")

        source = metrics.get("source")
        if isinstance(source, str) and source.strip():
            tokens.append(f"source={source.strip()}")

        doc = metrics.get("doc")
        if isinstance(doc, str) and doc.strip():
            tokens.append(f"doc={json.dumps(doc.strip(), ensure_ascii=False)}")

        tokens.extend(
            [
                f"context_chars={int(metrics.get('context_chars', 0) or 0)}",
                f"prompt_chars={int(metrics.get('prompt_chars', 0) or 0)}",
                f"input_chars={int(metrics.get('input_chars', 0) or 0)}",
            ]
        )

        ctx.logger.info("Step: %s (%s)", path, ", ".join(tokens))

    def on_step_end(self, ctx: FlowContext, record: dict[str, Any]) -> None:
        ctx.steps.append(record)
        path = record.get("path", "<unknown>")
        record_type = record.get("type") or "chat"
        meta = record.get("meta") if isinstance(record.get("meta"), dict) else {}
        stage_id = meta.get("stage_id") if isinstance(meta, dict) else None
        source = meta.get("source") if isinstance(meta, dict) else None
        suffix_tokens: list[str] = []
        if isinstance(stage_id, str) and stage_id.strip():
            suffix_tokens.append(f"stage_id={stage_id.strip()}")
        if isinstance(source, str) and source.strip():
            suffix_tokens.append(f"source={source.strip()}")

        if record_type == "action":
            if suffix_tokens:
                ctx.logger.info("Completed action %s (%s)", path, ", ".join(suffix_tokens))
            else:
                ctx.logger.info("Completed action %s", path)
            return

        input_chars = record.get("input_chars", 0)
        response_chars = record.get("response_chars", len(str(record.get("response", ""))))
        metrics_tokens: list[str] = [
            f"input_chars={int(input_chars or 0)}",
            f"chars={int(response_chars or 0)}",
            *suffix_tokens,
        ]
        ctx.logger.info("Received response for %s (%s)", path, ", ".join(metrics_tokens))

    def on_step_error(self, ctx: FlowContext, path: str, step_name: str, exc: Exception) -> None:
        ctx.logger.error("Step failed: %s (%s)", path, exc)


class NullStepRecorder:
    def on_step_start(
        self,
        ctx: FlowContext,
        path: str,
        **metrics: Any,
    ) -> None:
        return

    def on_step_end(self, ctx: FlowContext, record: dict[str, Any]) -> None:
        return

    def on_step_error(self, ctx: FlowContext, path: str, step_name: str, exc: Exception) -> None:
        return


class ChatRunner:
    def __init__(self, *, ai_text: Any, recorder: StepRecorder | None = None):
        self._ai_text = ai_text
        self._recorder = recorder or DefaultStepRecorder()
        self._validate_recorder(self._recorder)

    def run(self, ctx: FlowContext, node: Node) -> None:
        root_name = self._effective_root_name(node)
        root_path = [root_name]

        produced_messages, last_assistant = self._execute_node(
            ctx, node, parent_messages=ctx.messages, path_segments=root_path, inherited_meta=None
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
                pipeline_node_type=self._node_type(node),
                pipeline_node_name=root_name,
            )
            raise

    def run_steps(self, ctx: FlowContext, steps: list[ChatStep]) -> None:
        root = Block(name="pipeline", merge="all_messages", nodes=list(steps))
        self.run(ctx, root)

    def run_step(self, ctx: FlowContext, step: ChatStep) -> str:
        step_name = step.name or "step_01"
        produced, last_assistant = self._execute_node(
            ctx, step, parent_messages=ctx.messages, path_segments=[step_name], inherited_meta=None
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
        if isinstance(node, ActionStep):
            return node.name or "action_01"
        return node.name or "step_01"

    def _effective_child_name(self, node: Node, *, index: int) -> str:
        if isinstance(node, ChatStep):
            return node.name or f"step_{index + 1:02d}"
        if isinstance(node, ActionStep):
            return node.name or f"action_{index + 1:02d}"
        return node.name or f"block_{index + 1:02d}"

    def _node_type(self, node: Node) -> str:
        if isinstance(node, Block):
            return "block"
        if isinstance(node, ActionStep):
            return "action"
        return "step"

    def _merge_meta(
        self, inherited_meta: dict[str, Any] | None, node_meta: dict[str, Any] | None
    ) -> dict[str, Any]:
        if not inherited_meta and not node_meta:
            return {}
        merged: dict[str, Any] = {}
        if inherited_meta:
            merged.update(inherited_meta)
        if node_meta:
            merged.update(node_meta)
        return merged

    def _infer_stage_id(self, path_segments: list[str]) -> str | None:
        if len(path_segments) >= 2 and path_segments[0] == "pipeline":
            candidate = str(path_segments[1] or "").strip()
            return candidate or None
        return None

    def _prompt_source(self, prompt: str | Callable[[FlowContext], str]) -> str | None:
        if not callable(prompt):
            return None
        return self._callable_source(prompt)

    def _action_source(self, fn: Callable[[FlowContext], Any]) -> str | None:
        return self._callable_source(fn)

    def _callable_source(self, fn: Any) -> str | None:
        if not callable(fn):
            return None
        module = getattr(fn, "__module__", None) or "<unknown_module>"
        qualname = (
            getattr(fn, "__qualname__", None)
            or getattr(fn, "__name__", None)
            or "<callable>"
        )
        return f"{module}.{qualname}"

    def _json_safe(self, value: Any, *, max_depth: int = 4, max_items: int = 25) -> Any:
        if max_depth <= 0:
            return "<max_depth>"
        if value is None or isinstance(value, (bool, int, float, str)):
            return value
        if isinstance(value, (list, tuple)):
            items = list(value)
            trimmed = items[:max_items]
            out = [
                self._json_safe(item, max_depth=max_depth - 1, max_items=max_items)
                for item in trimmed
            ]
            if len(items) > max_items:
                out.append(f"<{len(items) - max_items} more>")
            return out
        if isinstance(value, dict):
            out: dict[str, Any] = {}
            for idx, (k, v) in enumerate(value.items()):
                if idx >= max_items:
                    out["<more>"] = f"<{len(value) - max_items} more>"
                    break
                out[str(k)] = self._json_safe(v, max_depth=max_depth - 1, max_items=max_items)
            return out
        return repr(value)

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
        ctx: FlowContext,
        node: Node,
        *,
        parent_messages: MessageHandler,
        path_segments: list[str],
        inherited_meta: dict[str, Any] | None,
    ) -> tuple[list[dict[str, Any]], str | None]:
        if isinstance(node, ChatStep):
            return self._execute_step(
                ctx,
                node,
                parent_messages=parent_messages,
                path_segments=path_segments,
                inherited_meta=inherited_meta,
            )
        if isinstance(node, ActionStep):
            return self._execute_action(
                ctx,
                node,
                parent_messages=parent_messages,
                path_segments=path_segments,
                inherited_meta=inherited_meta,
            )
        return self._execute_block(
            ctx,
            node,
            parent_messages=parent_messages,
            path_segments=path_segments,
            inherited_meta=inherited_meta,
        )

    def _execute_step(
        self,
        ctx: FlowContext,
        step: ChatStep,
        *,
        parent_messages: MessageHandler,
        path_segments: list[str],
        inherited_meta: dict[str, Any] | None,
    ) -> tuple[list[dict[str, Any]], str]:
        pipeline_path = "/".join(path_segments)
        step_name = path_segments[-1] if path_segments else (step.name or "<unnamed>")
        try:
            record_meta = self._merge_meta(inherited_meta, step.meta)
            if "stage_id" not in record_meta:
                stage_id = self._infer_stage_id(path_segments)
                if stage_id:
                    record_meta["stage_id"] = stage_id
            if "source" not in record_meta:
                source = self._prompt_source(step.prompt)
                if source:
                    record_meta["source"] = source

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
                node_type="chat",
                stage_id=record_meta.get("stage_id"),
                source=record_meta.get("source"),
                doc=record_meta.get("doc"),
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
                "type": "chat",
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
            if record_meta:
                record["meta"] = self._json_safe(record_meta)

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

    def _execute_action(
        self,
        ctx: FlowContext,
        action: ActionStep,
        *,
        parent_messages: MessageHandler,
        path_segments: list[str],
        inherited_meta: dict[str, Any] | None,
    ) -> tuple[list[dict[str, Any]], str | None]:
        pipeline_path = "/".join(path_segments)
        action_name = path_segments[-1] if path_segments else (action.name or "<unnamed>")
        try:
            record_meta = self._merge_meta(inherited_meta, action.meta)
            if "stage_id" not in record_meta:
                stage_id = self._infer_stage_id(path_segments)
                if stage_id:
                    record_meta["stage_id"] = stage_id
            if "source" not in record_meta:
                source = self._action_source(action.fn)
                if source:
                    record_meta["source"] = source

            self._recorder.on_step_start(
                ctx,
                pipeline_path,
                node_type="action",
                stage_id=record_meta.get("stage_id"),
                source=record_meta.get("source"),
                doc=record_meta.get("doc"),
            )

            result = action.fn(ctx)
            if action.capture_key is not None:
                ctx.outputs[action.capture_key] = result

            record: dict[str, Any] = {
                "type": "action",
                "name": action_name,
                "path": pipeline_path,
                "created_at": utc_now_iso8601(),
            }
            if record_meta:
                record["meta"] = self._json_safe(record_meta)
            if result is not None:
                record["result"] = self._json_safe(result)

            self._recorder.on_step_end(ctx, record)
            return [], None
        except Exception as exc:
            try:
                self._recorder.on_step_error(ctx, pipeline_path, action_name, exc)
            except Exception:
                ctx.logger.exception(
                    "Step recorder failed during error handling for %s", pipeline_path
                )
            self._attach_pipeline_error(
                exc,
                pipeline_path=pipeline_path,
                pipeline_node_type="action",
                pipeline_node_name=action_name,
            )
            raise

    def _execute_block(
        self,
        ctx: FlowContext,
        block: Block,
        *,
        parent_messages: MessageHandler,
        path_segments: list[str],
        inherited_meta: dict[str, Any] | None,
    ) -> tuple[list[dict[str, Any]], str | None]:
        block_name = path_segments[-1] if path_segments else (block.name or "<unnamed>")
        pipeline_path = "/".join(path_segments)
        try:
            block_meta = self._merge_meta(inherited_meta, block.meta)
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
                    inherited_meta=block_meta,
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
                        pipeline_node_type=self._node_type(child),
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
        self, ctx: FlowContext, produced_messages: list[dict[str, Any]]
    ) -> str | None:
        for message in reversed(produced_messages):
            if message.get("role") == ctx.agent_role:
                return message.get("content")  # type: ignore[return-value]
        return None

    def _apply_merge(
        self,
        ctx: FlowContext,
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
