from __future__ import annotations

import logging
from typing import Any

from message_handling import MessageHandler

from main import enclave_consensus, enclave_opinion


def message_with_log(
    ai_text: Any,
    messages_send: MessageHandler,
    messages_log: MessageHandler,
    prompt: str,
    agent_role: str,
    user_role: str,
    logger: logging.Logger | None = None,
    step_name: str | None = None,
    **kwargs,
):
    label = step_name or prompt[:60]
    if logger:
        logger.info("Dispatching prompt: %s", label)
    messages_send.continue_messages(user_role, prompt)
    messages_log.continue_messages(user_role, prompt)
    if logger:
        prompt_chars = len(prompt)
        input_chars = sum(
            len(str(message.get("content", "")))
            for message in messages_send.messages
            if isinstance(message, dict)
        )
        logger.info(
            "Sending prompt for %s (prompt_chars=%d, input_chars=%d)",
            label,
            prompt_chars,
            input_chars,
        )
    try:
        response = ai_text.text_chat(messages_send.messages, **kwargs)
    except Exception:
        if logger:
            logger.exception("AI text chat failed for step %s", label)
        raise
    if logger:
        logger.info("Received response for %s (chars=%d)", label, len(str(response)))
    messages_send.continue_messages(agent_role, response)
    messages_log.continue_messages(agent_role, response)
    return messages_send, messages_log, response


def tot_enclave(
    ai_text: Any,
    messages_main: MessageHandler,
    messages_log: MessageHandler,
    prompt: str,
    agent_role: str,
    user_role: str,
    logger: logging.Logger | None = None,
    **kwargs,
):
    messages = messages_main.copy()
    messages, messages_log, _ = message_with_log(
        ai_text,
        messages,
        messages_log,
        prompt,
        agent_role,
        user_role,
        logger=logger,
        **kwargs,
    )
    messages, messages_log, _ = message_with_log(
        ai_text,
        messages,
        messages_log,
        enclave_opinion(),
        agent_role,
        user_role,
        logger=logger,
        **kwargs,
    )
    messages, messages_log, response = message_with_log(
        ai_text,
        messages,
        messages_log,
        enclave_consensus(),
        agent_role,
        user_role,
        logger=logger,
        **kwargs,
    )
    messages_main.continue_messages(agent_role, response)
    return messages_main, messages_log
