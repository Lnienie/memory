"""Utility functions used in our graph."""

import os

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI

DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"


def load_chat_model(fully_specified_name: str) -> BaseChatModel:
    """Load a chat model from a fully specified name.

    Args:
        fully_specified_name (str): String in the format 'provider/model'.
    """
    fully_specified_name = fully_specified_name.strip()
    if "/" in fully_specified_name:
        provider, model = fully_specified_name.split("/", maxsplit=1)
    else:
        provider, model = "dashscope", fully_specified_name
    if provider in {"aliyun", "dashscope", "qwen"}:
        api_key = os.environ.get("DASHSCOPE_API_KEY")
        if not api_key:
            msg = "DASHSCOPE_API_KEY is required when using an Aliyun DashScope model."
            raise ValueError(msg)
        return ChatOpenAI(
            model=model,
            api_key=api_key,
            base_url=os.environ.get("DASHSCOPE_BASE_URL", DASHSCOPE_BASE_URL),
        )
    return init_chat_model(model, model_provider=provider)
