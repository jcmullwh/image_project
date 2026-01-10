from __future__ import annotations

import copy


class MessageHandler:
    def __init__(self, system_prompt: str | None = None):
        if system_prompt:
            self.messages = self.generate_first_message(system_prompt)
        else:
            default_prompt = "You are a helpful AI assistant."
            self.messages = self.generate_first_message(default_prompt)

    def generate_first_message(self, system_prompt: str) -> list[dict[str, str]]:
        return [{"role": "system", "content": system_prompt}]

    def continue_messages(self, role: str, new_prompt: str) -> None:
        self.messages.append({"role": role, "content": new_prompt})

    def copy(self) -> "MessageHandler":
        return copy.deepcopy(self)
