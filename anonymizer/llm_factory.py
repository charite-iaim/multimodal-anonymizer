"""
LLM Factory for creating LLM instances based on configured provider.

Supports:
- Azure OpenAI
- Fireworks AI
- Poe API (OpenAI-compatible)
- OpenRouter (OpenAI-compatible)
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
        use_vision_model: If True, use vision-capable model (for Fireworks)

    Returns:
        BaseChatModel instance configured for the selected provider
    """
    temp = temperature if temperature is not None else config.temperature

    if config.llm_provider == "azure":
        return _create_azure_llm(
            config=config,
            temperature=temp,
            timeout=timeout,
            max_tokens=max_tokens,
            tools=tools,
            structured_output=structured_output,
        )
    elif config.llm_provider == "fireworks":
        return _create_fireworks_llm(
            config=config,
            temperature=temp,
            timeout=timeout,
            max_tokens=max_tokens,
            tools=tools,
            structured_output=structured_output,
            use_vision_model=use_vision_model,
        )
    elif config.llm_provider == "poe":
        return _create_poe_llm(
            config=config,
            temperature=temp,
            timeout=timeout,
            max_tokens=max_tokens,
            tools=tools,
            structured_output=structured_output,
            use_vision_model=use_vision_model,
        )
    elif config.llm_provider == "openrouter":
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


def _create_azure_llm(
    config: AnonymizerConfig,
    temperature: float,
    timeout: int,
    max_tokens: int,
    tools: Optional[List[Any]] = None,
    structured_output: Optional[Type[BaseModel]] = None,
) -> BaseChatModel:
    """Create Azure OpenAI LLM instance."""
    from langchain_openai import AzureChatOpenAI

    llm = AzureChatOpenAI(
        azure_deployment=config.azure_deployment_name,
        azure_endpoint=config.azure_endpoint,
        api_key=config.azure_api_key,
        api_version=config.azure_api_version,
        timeout=timeout,
        max_tokens=max_tokens,
    )

    if structured_output is not None:
        return llm.with_structured_output(structured_output)
    elif tools is not None:
        return llm.bind_tools(tools)

    return llm


def _create_fireworks_llm(
    config: AnonymizerConfig,
    temperature: float,
    timeout: int,
    max_tokens: int,
    tools: Optional[List[Any]] = None,
    structured_output: Optional[Type[BaseModel]] = None,
    use_vision_model: bool = False,
) -> BaseChatModel:
    """Create Fireworks AI LLM instance."""
    from langchain_fireworks import ChatFireworks

    model = config.fireworks_vision_model if use_vision_model else config.fireworks_model

    llm = ChatFireworks(
        model=model,
        api_key=config.fireworks_api_key,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
    )

    if structured_output is not None:
        return llm.with_structured_output(structured_output)
    elif tools is not None:
        return llm.bind_tools(tools)

    return llm


def _create_poe_llm(
    config: AnonymizerConfig,
    temperature: float,
    timeout: int,
    max_tokens: int,
    tools: Optional[List[Any]] = None,
    structured_output: Optional[Type[BaseModel]] = None,
    use_vision_model: bool = False,
) -> BaseChatModel:
    """Create Poe API LLM instance (OpenAI-compatible)."""
    from langchain_openai import ChatOpenAI

    model = config.poe_vision_model if use_vision_model else config.poe_model

    llm = ChatOpenAI(
        model=model,
        api_key=config.poe_api_key,
        base_url=config.poe_base_url,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
    )

    if structured_output is not None:
        return llm.with_structured_output(structured_output)
    elif tools is not None:
        return llm.bind_tools(tools)

    return llm


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

    # Use vision model if specified and requested, otherwise use the main model
    if use_vision_model and config.local_vision_model:
        model = config.local_vision_model
    else:
        model = config.local_model

    # Use a dummy key if none provided (most local servers don't need authentication)
    api_key = config.local_api_key or "not-needed"

    llm = ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=config.local_base_url,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
    )

    if structured_output is not None:
        return llm.with_structured_output(structured_output)
    elif tools is not None:
        return llm.bind_tools(tools)

    return llm