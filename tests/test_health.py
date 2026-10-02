import pytest

@pytest.mark.asyncio
async def test_health_check(async_client):
    response = await async_client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "database" in data
    assert "ollama" in data
    assert "embedding_model" in data
    assert data["status"] == "ok"
    assert data["database"] == "ok"
    assert data["ollama"] in ["ok", "unavailable"]
    assert data["embedding_model"] == "BAAI/bge-small-en-v1.5"
