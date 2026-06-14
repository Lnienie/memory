"""Graphs that extract memories on a schedule."""

import asyncio
import logging
from datetime import datetime
from typing import Any, Awaitable, Callable, cast

from langgraph.graph import END, StateGraph
from langgraph.runtime import Runtime
from langgraph.store.base import BaseStore

from memory_agent import tools, utils
from memory_agent.context import Context
from memory_agent.state import State

logger = logging.getLogger(__name__)

MEMORY_TOOLS: dict[
    str,
    Callable[..., Awaitable[str]],
] = {
    "upsert_memory": tools.upsert_memory,
    "upsert_knowledge": tools.upsert_knowledge,
    "upsert_experience": tools.upsert_experience,
}


async def _search_memories(state: State, runtime: Runtime[Context]) -> dict[str, list[Any]]:
    """Retrieve relevant user, knowledge, and experience memories."""
    store = cast(BaseStore, runtime.store)
    query = str([m.content for m in state.messages[-3:]])
    user_id = runtime.context.user_id
    user_memories, knowledge_memories, experience_memories = await asyncio.gather(
        store.asearch(tools.user_memory_namespace(user_id), query=query, limit=10),
        store.asearch(tools.knowledge_namespace(), query=query, limit=10),
        store.asearch(tools.experience_namespace(), query=query, limit=10),
    )
    return {
        "用户记忆": user_memories,
        "知识库": knowledge_memories,
        "经验库": experience_memories,
    }


def _format_memory_value(value: Any) -> str:
    """Format a stored memory payload for prompt injection."""
    if not isinstance(value, dict):
        return str(value)

    lines: list[str] = []
    for key, item in value.items():
        if item in (None, "", {}, []):
            continue
        lines.append(f"{key}: {item}")
    return "; ".join(lines) if lines else str(value)


def _format_memory_sections(memory_groups: dict[str, list[Any]]) -> str:
    """Render retrieved memories into a single prompt block."""
    sections: list[str] = []
    for label, memories in memory_groups.items():
        if not memories:
            continue
        body = "\n".join(
            f"- [{mem.key}] {_format_memory_value(mem.value)} (similarity: {mem.score})"
            for mem in memories
        )
        sections.append(f"<{label}>\n{body}\n</{label}>")
    return "\n\n".join(sections)


async def call_model(state: State, runtime: Runtime[Context]) -> dict:
    """Extract the user's state from the conversation and update the memory."""
    model = runtime.context.model
    system_prompt = runtime.context.system_prompt

    memory_groups = await _search_memories(state, runtime)
    formatted_memories = _format_memory_sections(memory_groups)

    sys = system_prompt.format(
        user_info=formatted_memories,
        memory_context=formatted_memories,
        time=datetime.now().isoformat(),
    )

    llm = utils.load_chat_model(model)
    msg = await llm.bind_tools(list(MEMORY_TOOLS.values())).ainvoke(
        [{"role": "system", "content": sys}, *state.messages]
    )
    return {"messages": [msg]}


async def _execute_tool_call(tool_call: dict[str, Any], runtime: Runtime[Context]) -> str:
    """Execute a supported memory tool call."""
    tool_name = tool_call["name"]
    tool = MEMORY_TOOLS[tool_name]
    tool_args = dict(tool_call.get("args", {}))
    if tool_name == "upsert_memory":
        tool_args["user_id"] = runtime.context.user_id
    tool_args["store"] = cast(BaseStore, runtime.store)
    return await tool(**tool_args)


async def store_memory(state: State, runtime: Runtime[Context]):
    """Persist supported memory tool calls emitted by the model."""
    tool_calls = getattr(state.messages[-1], "tool_calls", [])
    supported_calls = [tc for tc in tool_calls if tc.get("name") in MEMORY_TOOLS]

    saved_memories = await asyncio.gather(
        *(_execute_tool_call(tc, runtime) for tc in supported_calls)
    )

    results = [
        {
            "role": "tool",
            "content": mem,
            "tool_call_id": tc["id"],
        }
        for tc, mem in zip(supported_calls, saved_memories)
    ]
    return {"messages": results}


def route_message(state: State):
    """Determine the next step based on the presence of tool calls."""
    msg = state.messages[-1]
    if any(tc.get("name") in MEMORY_TOOLS for tc in getattr(msg, "tool_calls", [])):
        return "store_memory"
    return END


builder = StateGraph(State, context_schema=Context)
builder.add_node(call_model)
builder.add_edge("__start__", "call_model")
builder.add_node(store_memory)
builder.add_conditional_edges("call_model", route_message, ["store_memory", END])
builder.add_edge("store_memory", "call_model")
graph = builder.compile()
graph.name = "MemoryAgent"


__all__ = ["graph"]
