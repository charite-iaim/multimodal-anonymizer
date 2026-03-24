"""
LLM Factory for creating LLM instances based on configured provider.

Supports:
- OpenRouter (OpenAI-compatible)
- Local LLM (OpenAI-compatible: Ollama, LM Studio, vLLM, LocalAI, etc.)
"""

from typing import Optional, List, Any, Type
from pydantic import BaseModel

from langchain_core.language_models.chat_models import BaseChatModel

from .config import AnonymizerConfig


def create_chat_llm(
    config: AnonymizerConfig,
    temperature: Optional[float] = None,
    timeout: int = 600,
    max_tokens: int = 16000,
    tools: Optional[List[Any]] = None,
    structured_output: Optional[Type[BaseModel]] = None,
    use_vision_model: bool = False,
) -> BaseChatModel:
    """
    Create a chat LLM instance based on the configured provider.

    Args:
        config: Anonymizer configuration with provider settings
        temperature: Override temperature (uses config.temperature if None)
        timeout: Request timeout in seconds
        max_tokens: Maximum tokens in response
        tools: Optional list of tools to bind to the LLM
        structured_output: Optional Pydantic model for structured output
        use_vision_model: If True, use vision-capable model

    Returns:
        BaseChatModel instance configured for the selected provider
    """
    temp = temperature if temperature is not None else config.temperature

    if config.llm_provider == "openrouter":
        return _create_openrouter_llm(
            config=config,
            temperature=temp,
            timeout=timeout,
            max_tokens=max_tokens,
            tools=tools,
            structured_output=structured_output,
            use_vision_model=use_vision_model,
        )
    elif config.llm_provider == "local":
        return _create_local_llm(
            config=config,
            temperature=temp,
            timeout=timeout,
            max_tokens=max_tokens,
            tools=tools,
            structured_output=structured_output,
            use_vision_model=use_vision_model,
        )
    else:
        raise ValueError(f"Unsupported LLM provider: {config.llm_provider}")


def _create_openrouter_llm(
    config: AnonymizerConfig,
    temperature: float,
    timeout: int,
    max_tokens: int,
    tools: Optional[List[Any]] = None,
    structured_output: Optional[Type[BaseModel]] = None,
    use_vision_model: bool = False,
) -> BaseChatModel:
    """Create OpenRouter LLM instance (OpenAI-compatible)."""
    from langchain_openai import ChatOpenAI

    model = config.openrouter_vision_model if use_vision_model else config.openrouter_model

    llm = ChatOpenAI(
        model=model,
        api_key=config.openrouter_api_key,
        base_url=config.openrouter_base_url,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
    )

    if structured_output is not None:
        return llm.with_structured_output(structured_output)
    elif tools is not None:
        return llm.bind_tools(tools)

    return llm


def _create_local_llm(
    config: AnonymizerConfig,
    temperature: float,
    timeout: int,
    max_tokens: int,
    tools: Optional[List[Any]] = None,
    structured_output: Optional[Type[BaseModel]] = None,
    use_vision_model: bool = False,
) -> BaseChatModel:
    """
    Create local LLM instance (OpenAI-compatible).

    Works with any local LLM server that provides an OpenAI-compatible API:
    - Ollama (http://localhost:11434/v1)
    - LM Studio (http://localhost:1234/v1)
    - vLLM (http://localhost:8000/v1)
    - LocalAI (http://localhost:8080/v1)
    - text-generation-webui with OpenAI extension
    - Any other OpenAI-compatible server
    """
    from langchain_openai import ChatOpenAI
    import httpx

    # Use vision model if specified and requested, otherwise use the main model
    if use_vision_model and config.local_vision_model:
        model = config.local_vision_model
    else:
        model = config.local_model

    # Use a dummy key if none provided
    api_key = config.local_api_key or "not-needed"

    # Disable SSL verification for localhost tunnels
    http_client = None
    if config.local_base_url and "localhost" in config.local_base_url:
        http_client = httpx.Client(verify=False)

    # Pass thinking mode via extra_body for reasoning models
    model_kwargs = {}
    if not config.local_thinking:
        model_kwargs["extra_body"] = {
            "chat_template_kwargs": {"thinking": False}
        }

    llm = ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=config.local_base_url,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
        **({"http_client": http_client} if http_client else {}),
        **model_kwargs,
    )

    if structured_output is not None:
        return llm.with_structured_output(structured_output)
    elif tools is not None:
        return llm.bind_tools(tools)

    return llm