"""Define the agent's memory storage tools."""

import uuid
from typing import Annotated, Any, TypedDict

from langchain_core.tools import InjectedToolArg
from langgraph.store.base import BaseStore

StoreNamespace = tuple[str, ...]

USER_MEMORY_COLLECTION = "memories"
KNOWLEDGE_COLLECTION = "knowledge"
EXPERIENCE_COLLECTION = "experiences"


class UserMemoryValue(TypedDict):
    """Represent the stored payload for user memory records."""

    content: str
    context: str


class KnowledgeMemoryValue(TypedDict):
    """Represent the stored payload for knowledge records."""

    content: str
    context: str
    title: str | None
    source: str | None
    metadata: dict[str, Any]


class ExperienceMemoryValue(TypedDict):
    """Represent the stored payload for reusable experience records."""

    content: str
    context: str
    problem: str | None
    solution: str | None
    metadata: dict[str, Any]


def user_memory_namespace(user_id: str) -> StoreNamespace:
    """Return the namespace for user-scoped conversational memories."""
    return (USER_MEMORY_COLLECTION, user_id)


def knowledge_namespace() -> StoreNamespace:
    """Return the namespace for shared knowledge records."""
    return (KNOWLEDGE_COLLECTION,)


def experience_namespace() -> StoreNamespace:
    """Return the namespace for shared reusable experience records."""
    return (EXPERIENCE_COLLECTION,)


async def _upsert_store_value(
    *,
    store: BaseStore,
    namespace: StoreNamespace,
    value: dict[str, Any],
    memory_id: uuid.UUID | None,
) -> uuid.UUID:
    """Persist a record to the configured store and return its identifier."""
    mem_id = memory_id or uuid.uuid4()
    await store.aput(namespace, key=str(mem_id), value=value)
    return mem_id


async def upsert_memory(
    content: str,
    context: str,
    *,
    memory_id: uuid.UUID | None = None,
    user_id: Annotated[str, InjectedToolArg],
    store: Annotated[BaseStore, InjectedToolArg],
):
    """Upsert a user memory in the database.

    If a memory conflicts with an existing one, then just UPDATE the
    existing one by passing in memory_id - don't create two memories
    that are the same. If the user corrects a memory, UPDATE it.

    Args:
        content: The main content of the memory. For example:
            "User expressed interest in learning about French."
        context: Additional context for the memory. For example:
            "This was mentioned while discussing career options in Europe."
        memory_id: ONLY PROVIDE IF UPDATING AN EXISTING MEMORY.
        The memory to overwrite.
    """
    value: UserMemoryValue = {"content": content, "context": context}
    mem_id = await _upsert_store_value(
        store=store,
        namespace=user_memory_namespace(user_id),
        value=value,
        memory_id=memory_id,
    )
    return f"Stored memory {mem_id}"


async def upsert_knowledge(
    content: str,
    context: str,
    *,
    title: str | None = None,
    source: str | None = None,
    metadata: dict[str, Any] | None = None,
    knowledge_id: uuid.UUID | None = None,
    store: Annotated[BaseStore, InjectedToolArg],
):
    """Upsert a knowledge record in the shared knowledge store."""
    value: KnowledgeMemoryValue = {
        "content": content,
        "context": context,
        "title": title,
        "source": source,
        "metadata": metadata or {},
    }
    mem_id = await _upsert_store_value(
        store=store,
        namespace=knowledge_namespace(),
        value=value,
        memory_id=knowledge_id,
    )
    return f"Stored knowledge {mem_id}"


async def upsert_experience(
    content: str,
    context: str,
    *,
    problem: str | None = None,
    solution: str | None = None,
    metadata: dict[str, Any] | None = None,
    experience_id: uuid.UUID | None = None,
    store: Annotated[BaseStore, InjectedToolArg],
):
    """Upsert a reusable experience record in the shared experience store."""
    value: ExperienceMemoryValue = {
        "content": content,
        "context": context,
        "problem": problem,
        "solution": solution,
        "metadata": metadata or {},
    }
    mem_id = await _upsert_store_value(
        store=store,
        namespace=experience_namespace(),
        value=value,
        memory_id=experience_id,
    )
    return f"Stored experience {mem_id}"


__all__ = [
    "EXPERIENCE_COLLECTION",
    "KNOWLEDGE_COLLECTION",
    "USER_MEMORY_COLLECTION",
    "ExperienceMemoryValue",
    "KnowledgeMemoryValue",
    "StoreNamespace",
    "UserMemoryValue",
    "experience_namespace",
    "knowledge_namespace",
    "upsert_experience",
    "upsert_knowledge",
    "upsert_memory",
    "user_memory_namespace",
]
