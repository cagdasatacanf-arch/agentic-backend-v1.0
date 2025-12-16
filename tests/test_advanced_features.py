
import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from fastapi.testclient import TestClient

from app.main import app
from app.utils.chunking import TextChunker
from app.services.agent_service import agent_service

client = TestClient(app)

# --- Chunking Tests ---

def test_chunking_strategies():
    chunker = TextChunker(chunk_size=50, chunk_overlap=10)
    
    # 1. Text shorter than chunk size
    short_text = "Short text."
    chunks = chunker.chunk_text(short_text)
    assert len(chunks) == 1
    assert chunks[0]["text"] == short_text
    
    # 2. Text longer than chunk size
    long_text = "This is a longer text that definitely exceeds the fifty character limit set for this test case."
    chunks = chunker.chunk_text(long_text)
    assert len(chunks) > 1
    # Check overlap (roughly)
    # This is a heuristic check; exact overlap depends on logic
    
    # 3. Empty text
    assert len(chunker.chunk_text("")) == 1
    assert chunker.chunk_text("")[0]["text"] == ""

def test_chunk_by_sentences():
    chunker = TextChunker(chunk_size=50)
    text = "Sentence one. Sentence two. Sentence three is longer."
    chunks = chunker.chunk_by_sentences(text)
    assert len(chunks) >= 1
    for chunk in chunks:
        assert "text" in chunk
        assert "metadata" in chunk

# --- Tool Tests ---

def test_calculator_tool_complex():
    # Valid complex operation
    res = agent_service._calculator_tool("(10 + 2) * 3 / 4")
    assert float(res) == 9.0
    
    # Invalid characters
    res = agent_service._calculator_tool("import os")
    assert "Error" in res
    
    # Division by zero
    res = agent_service._calculator_tool("1/0")
    assert "Error" in res

# --- Session Memory Tests ---

@pytest.mark.asyncio
async def test_memory_window_limit():
    session_id = "memory_stress_test"
    
    # Mock redis
    agent_service.redis = MagicMock()
    
    # Add 25 messages
    for i in range(25):
        agent_service._add_to_history(session_id, "user", f"msg {i}")
        
    # Check if ltrim was called 25 times
    assert agent_service.redis.ltrim.call_count == 25
    # Check if we trimmed to correct size (-20, -1)
    agent_service.redis.ltrim.assert_called_with(f"session:{session_id}", -20, -1)

# --- Rate Limit Test ---
# Note: Testing slowapi with TestClient can be tricky as it requires mocking remote IP
# We will simulate the limiter logic or check if app has limiter state

def test_limiter_configured():
    assert hasattr(app.state, "limiter")
    # We can check specific route limits if we inspect the router, 
    # but presence of limiter on app state confirms initialization.
