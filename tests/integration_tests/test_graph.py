from types import SimpleNamespace
from typing import List

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END
from langgraph.store.memory import InMemoryStore

from memory_agent import tools
from memory_agent.context import Context
from memory_agent.graph import (
    _format_memory_sections,
    _search_memories,
    builder,
    route_message,
)
from memory_agent.state import State

pytestmark = pytest.mark.anyio


@pytest.mark.parametrize(
    "conversation",
    [
        ["My name is Alice and I love pizza. Remember this."],
        [
            "Hi, I'm Bob and I enjoy playing tennis. Remember this.",
            "Yes, I also have a pet dog named Max.",
            "Max is a golden retriever and he's 5 years old. Please remember this too.",
        ],
        [
            "Hello, I'm Charlie. I work as a software engineer and I'm passionate about AI. Remember this.",
            "I specialize in machine learning algorithms and I'm currently working on a project involving natural language processing.",
            "My main goal is to improve sentiment analysis accuracy in multi-lingual texts. It's challenging but exciting.",
            "We've made some progress using transformer models, but we're still working on handling context and idioms across languages.",
            "Chinese and English have been the most challenging pair so far due to their vast differences in structure and cultural contexts.",
        ],
    ],
    ids=["short", "medium", "long"],
)
async def test_memory_storage(conversation: List[str]):
    mem_store = InMemoryStore()

    graph = builder.compile(store=mem_store, checkpointer=InMemorySaver())
    user_id = "test-user"

    for content in conversation:
        await graph.ainvoke(
            {"messages": [("user", content)]},
            {"thread_id": "thread"},
            context=Context(user_id=user_id),
        )

    namespace = tools.user_memory_namespace(user_id)
    memories = mem_store.search(namespace)

    assert len(memories) > 0

    bad_namespace = tools.user_memory_namespace("wrong-user")
    bad_memories = mem_store.search(bad_namespace)
    assert len(bad_memories) == 0


async def test_knowledge_and_experience_storage():
    mem_store = InMemoryStore()

    await tools.upsert_knowledge(
        content="PostgreSQL uses MVCC to provide transaction isolation.",
        context="Database internals",
        title="PostgreSQL MVCC",
        source="internal-wiki",
        metadata={"domain": "database"},
        store=mem_store,
    )
    await tools.upsert_experience(
        content="When async tests hang, check for un-awaited coroutines first.",
        context="Python test troubleshooting",
        problem="Async pytest suite stalls",
        solution="Inspect warnings and await spawned tasks.",
        metadata={"language": "python"},
        store=mem_store,
    )

    knowledge_records = mem_store.search(tools.knowledge_namespace())
    experience_records = mem_store.search(tools.experience_namespace())

    assert len(knowledge_records) == 1
    assert len(experience_records) == 1
    assert knowledge_records[0].value["title"] == "PostgreSQL MVCC"
    assert experience_records[0].value["problem"] == "Async pytest suite stalls"


async def test_search_memories_combines_user_knowledge_and_experience():
    mem_store = InMemoryStore()
    user_id = "retrieval-user"

    await tools.upsert_memory(
        content="User prefers PostgreSQL for transactional systems.",
        context="Architecture discussion",
        user_id=user_id,
        store=mem_store,
    )
    await tools.upsert_knowledge(
        content="PostgreSQL uses MVCC to provide transaction isolation.",
        context="Database internals",
        store=mem_store,
    )
    await tools.upsert_experience(
        content="Reuse MVCC guidance when diagnosing lock contention.",
        context="Incident response",
        problem="Database lock contention",
        solution="Inspect long-running transactions before tuning indexes.",
        store=mem_store,
    )

    runtime = SimpleNamespace(context=Context(user_id=user_id), store=mem_store)
    state = State(messages=[HumanMessage(content="How does PostgreSQL help with transactions?")])

    memory_groups = await _search_memories(state, runtime)

    assert len(memory_groups["用户记忆"]) == 1
    assert len(memory_groups["知识库"]) == 1
    assert len(memory_groups["经验库"]) == 1


async def test_search_memories_handles_empty_knowledge_and_experience():
    mem_store = InMemoryStore()
    user_id = "memory-only-user"

    await tools.upsert_memory(
        content="User cares about concise answers.",
        context="Response preferences",
        user_id=user_id,
        store=mem_store,
    )

    runtime = SimpleNamespace(context=Context(user_id=user_id), store=mem_store)
    state = State(messages=[HumanMessage(content="Keep it brief.")])

    memory_groups = await _search_memories(state, runtime)
    rendered = _format_memory_sections(memory_groups)

    assert len(memory_groups["用户记忆"]) == 1
    assert memory_groups["知识库"] == []
    assert memory_groups["经验库"] == []
    assert "<用户记忆>" in rendered
    assert "<知识库>" not in rendered
    assert "<经验库>" not in rendered


def test_route_message_only_handles_supported_memory_tools():
    supported_state = State(
        messages=[AIMessage(content="", tool_calls=[{"name": "upsert_knowledge", "args": {}, "id": "1"}])]
    )
    unsupported_state = State(
        messages=[AIMessage(content="", tool_calls=[{"name": "web_search", "args": {}, "id": "2"}])]
    )

    assert route_message(supported_state) == "store_memory"
    assert route_message(unsupported_state) == END
