from app.services.rag.rerank import local_rerank_documents


class _FakeCrossEncoder:
    def predict(self, pairs, **_kwargs):
        assert pairs[0][0] == "token refresh"
        assert "full source content" in pairs[0][1]
        return [0.9, 0.2]


def test_local_reranker_uses_raw_content_and_records_model(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.services.rag.rerank._local_cross_encoder",
        lambda _model, _device: _FakeCrossEncoder(),
    )
    candidates = [
        {
            "id": "a",
            "title": "auth.md",
            "snippet": "short snippet",
            "_raw_content": "full source content about token refresh",
            "metadata": {"path": "docs/auth.md"},
            "retrieval": {},
        },
        {
            "id": "b",
            "title": "other.md",
            "snippet": "unrelated",
            "metadata": {},
            "retrieval": {},
        },
    ]

    results = local_rerank_documents("token refresh", candidates, 2, model_name="local/model")

    assert [item["id"] for item in results] == ["a", "b"]
    assert results[0]["retrieval"]["rerank_mode"] == "local"
    assert results[0]["retrieval"]["rerank_model"] == "local/model"
