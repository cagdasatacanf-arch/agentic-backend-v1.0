
import pytest
from app.utils.chunking import default_chunker
from app.services.agent_service import agent_service

@pytest.mark.asyncio
async def test_chunker():
    text = "Hello world. " * 300
    chunks = default_chunker.chunk_text(text)
    assert len(chunks) > 1
    assert "metadata" in chunks[0]
    assert chunks[0]["metadata"]["chunk_index"] == 0

@pytest.mark.asyncio
async def test_tools():
    # Test calculator tool
    result = agent_service._calculator_tool("2 + 2")
    assert result == "4"
    
    result = agent_service._calculator_tool("10 * 5")
    assert result == "50"
    
    # Test safe eval
    result = agent_service._calculator_tool("__import__('os').system('ls')")
    assert "Error" in result

@pytest.mark.asyncio
async def test_session_memory():
    # Mock redis for this test
    # We can't test actual redis logic without a running redis, 
    # so we verify the method is called 
    from unittest.mock import MagicMock
    agent_service.redis = MagicMock()
    
    session_id = "test-session-123"
    
    # Simulate adding to history
    agent_service._add_to_history(session_id, "user", "Hello")
    
    # Assert rpush was called
    agent_service.redis.rpush.assert_called()
