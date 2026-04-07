import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from msagent.cli.bootstrap.initializer import Initializer
from msagent.configs import CheckpointerConfig, CheckpointerProvider


@pytest.mark.asyncio
async def test_memory_checkpointer_factory_returns_memory_saver() -> None:
    initializer = Initializer()
    config = CheckpointerConfig(type=CheckpointerProvider.MEMORY)

    async with initializer._create_checkpointer(config, None) as checkpointer:
        assert isinstance(checkpointer, InMemorySaver)


@pytest.mark.asyncio
async def test_sqlite_checkpointer_factory_returns_async_sqlite_saver(tmp_path) -> None:
    initializer = Initializer()
    config = CheckpointerConfig(type=CheckpointerProvider.SQLITE)
    db_path = tmp_path / "checkpoints.db"

    async with initializer._create_checkpointer(config, str(db_path)) as checkpointer:
        assert isinstance(checkpointer, AsyncSqliteSaver)

