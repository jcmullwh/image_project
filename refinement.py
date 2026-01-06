from __future__ import annotations

from typing import Any, Callable

from pipeline import Block, ChatStep, RunContext


class RefinementPolicy:
    def stage(
        self,
        stage_name: str,
        *,
        prompt: str | Callable[[RunContext], str],
        temperature: float,
        allow_empty_prompt: bool = False,
        allow_empty_response: bool = False,
        params: dict[str, Any] | None = None,
        capture_key: str | None = None,
    ) -> Block:
        raise NotImplementedError

    def _validated_stage_name(self, stage_name: str) -> str:
        if not isinstance(stage_name, str):
            raise TypeError(f"Stage name must be a string (type={type(stage_name).__name__})")
        trimmed = stage_name.strip()
        if not trimmed:
            raise ValueError("Stage name cannot be empty")
        return trimmed

    def _validated_prompt(self, prompt: str | Callable[[RunContext], str]) -> str | Callable[[RunContext], str]:
        if isinstance(prompt, str) or callable(prompt):
            return prompt
        raise TypeError(
            f"Stage prompt must be a string or callable (type={type(prompt).__name__})"
        )

    def _validated_params(self, params: dict[str, Any] | None) -> dict[str, Any]:
        if params is None:
            return {}
        if not isinstance(params, dict):
            raise TypeError(
                f"Stage params must be a dict if provided (type={type(params).__name__})"
            )
        return dict(params)


class NoRefinement(RefinementPolicy):
    def stage(
        self,
        stage_name: str,
        *,
        prompt: str | Callable[[RunContext], str],
        temperature: float,
        allow_empty_prompt: bool = False,
        allow_empty_response: bool = False,
        params: dict[str, Any] | None = None,
        capture_key: str | None = None,
    ) -> Block:
        stage_name = self._validated_stage_name(stage_name)
        prompt = self._validated_prompt(prompt)
        params = self._validated_params(params)

        draft_step = ChatStep(
            name="draft",
            prompt=prompt,
            temperature=temperature,
            allow_empty_prompt=allow_empty_prompt,
            allow_empty_response=allow_empty_response,
            params=params,
        )
        return Block(
            name=stage_name,
            merge="last_response",
            nodes=[draft_step],
            capture_key=capture_key,
        )


class TotEnclaveRefinement(RefinementPolicy):
    def stage(
        self,
        stage_name: str,
        *,
        prompt: str | Callable[[RunContext], str],
        temperature: float,
        allow_empty_prompt: bool = False,
        allow_empty_response: bool = False,
        params: dict[str, Any] | None = None,
        capture_key: str | None = None,
    ) -> Block:
        stage_name = self._validated_stage_name(stage_name)
        prompt = self._validated_prompt(prompt)
        params = self._validated_params(params)

        # Local import to avoid circular dependency.
        from refinement_enclave import make_tot_enclave_block

        draft_step = ChatStep(
            name="draft",
            prompt=prompt,
            temperature=temperature,
            allow_empty_prompt=allow_empty_prompt,
            allow_empty_response=allow_empty_response,
            params=params,
        )
        enclave_block = make_tot_enclave_block(stage_name)
        return Block(
            name=stage_name,
            merge="last_response",
            nodes=[draft_step, enclave_block],
            capture_key=capture_key,
        )
