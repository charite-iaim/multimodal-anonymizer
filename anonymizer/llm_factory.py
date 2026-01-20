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
    reasoning_effort: Optional[str] = "medium",
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
        reasoning_effort: Reasoning effort for reasoning models ("none", "low", "medium", "high").
                          Set to None to disable reasoning. Default is "medium".

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
            reasoning_effort=reasoning_effort,
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
    else:
        raise ValueError(f"Unsupported LLM provider: {config.llm_provider}")


def _create_azure_llm(
    config: AnonymizerConfig,
    temperature: float,
    timeout: int,
    max_tokens: int,
    tools: Optional[List[Any]] = None,
    structured_output: Optional[Type[BaseModel]] = None,
    reasoning_effort: Optional[str] = "medium",
) -> BaseChatModel:
    """Create Azure OpenAI LLM instance."""
    from langchain_openai import AzureChatOpenAI

    # Build model kwargs for reasoning effort
    model_kwargs = {}
    if reasoning_effort is not None:
        model_kwargs["reasoning_effort"] = reasoning_effort

    llm_kwargs = {
        "azure_deployment": config.azure_deployment_name,
        "azure_endpoint": config.azure_endpoint,
        "api_key": config.azure_api_key,
        "api_version": config.azure_api_version,
        "timeout": timeout,
        "max_tokens": max_tokens,
    }
    # Note: temperature is intentionally not passed to Azure

    if model_kwargs:
        llm_kwargs["model_kwargs"] = model_kwargs

    llm = AzureChatOpenAI(**llm_kwargs)

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