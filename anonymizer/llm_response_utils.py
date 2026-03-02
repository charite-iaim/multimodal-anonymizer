"""
Utilities for parsing LLM responses from reasoning/thinking models.

Reasoning models (DeepSeek, Qwen3-Thinking, QwQ, GLM-4, kimi-k2.5, etc.) may
include chain-of-thought text in their responses. This module provides helpers
to strip that reasoning and extract only the actual answer.

Thinking content can appear in several forms:
- Wrapped in <think>...</think> tags (DeepSeek, some Qwen3 configs)
- As a separate `reasoning_content` field in the API response (OpenAI-compatible APIs)
- As raw inline text before the actual answer (Qwen3-Thinking without tags)
"""

import re


def strip_thinking_content(raw_text: str) -> str:
    """
    Strip chain-of-thought / reasoning blocks from an LLM response.

    Handles:
    1. <think>...</think> tags (case-insensitive, may span multiple lines)
    2. Orphaned </think> closing tags (when the opening tag is missing or was
       consumed by the API as `reasoning_content`)
    3. ```think ... ``` code blocks (some models use markdown-style thinking)

    Args:
        raw_text: Raw LLM response content.

    Returns:
        The response with thinking blocks removed. If stripping leaves an
        empty string, returns the original text unchanged.
    """
    if not raw_text:
        return raw_text

    cleaned = raw_text

    # 1. Strip paired <think>...</think> blocks
    cleaned = re.sub(r'<think>.*?</think>', '', cleaned, flags=re.DOTALL | re.IGNORECASE)

    # 2. Strip everything before an orphaned </think> tag (model returned
    #    reasoning without an opening tag — the opening was either absent or
    #    the API split it into a separate field)
    cleaned = re.sub(r'^.*?</think>\s*', '', cleaned, flags=re.DOTALL | re.IGNORECASE)

    cleaned = cleaned.strip()

    # If nothing meaningful remains, fall back to original
    if not cleaned:
        return raw_text

    return cleaned


def extract_content_from_response(response) -> str:
    """
    Extract the actual content from a LangChain LLM response, stripping
    any reasoning/thinking content.

    Handles:
    - Standard `.content` attribute
    - `reasoning_content` in `additional_kwargs` (used by some OpenAI-compatible
      APIs for reasoning models)

    Args:
        response: A LangChain BaseMessage (AIMessage, etc.)

    Returns:
        Cleaned content string with thinking blocks removed.
    """
    # Get the main content
    if hasattr(response, 'content'):
        content = response.content
    else:
        content = str(response)

    # Some APIs separate reasoning into additional_kwargs.reasoning_content,
    # in which case the main content is already clean. But strip just in case.
    return strip_thinking_content(content)


def get_reasoning_content_from_response(response) -> str:
    """
    Extract reasoning/thinking content from a LangChain LLM response.

    Reasoning models may expose their chain-of-thought via:
    - `additional_kwargs.reasoning_content` (OpenAI-compatible APIs)
    - `<think>...</think>` tags inside `.content`

    Args:
        response: A LangChain BaseMessage (AIMessage, etc.)

    Returns:
        The reasoning/thinking text, or empty string if none found.
    """
    # Check additional_kwargs first (some APIs separate reasoning there)
    additional = getattr(response, 'additional_kwargs', {})
    if 'reasoning_content' in additional and additional['reasoning_content']:
        return str(additional['reasoning_content'])

    # Fall back to extracting <think> blocks from content
    content = getattr(response, 'content', '') or ''
    think_matches = re.findall(r'<think>(.*?)</think>', content, flags=re.DOTALL | re.IGNORECASE)
    if think_matches:
        return '\n'.join(think_matches).strip()

    return ''
