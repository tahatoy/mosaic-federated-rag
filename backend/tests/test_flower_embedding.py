import pytest
import numpy as np
from unittest.mock import MagicMock
from client_setup.flower_rag_client import EmbeddingFlowerClient

@pytest.fixture
def mock_rag_engine():
    engine = MagicMock()
    # Mock ChromaDB collection
    engine.collection.get.return_value = {
        "documents": [
            "Bu birinci cümledir. Bu da devamıdır. Bir şeyler daha ve daha da fazlası.",
            "Kısa döküman.",
            "Bu da başka bir döküman. Finansal veriler içerir ve oldukça uzundur kendisi."
        ]
    }
    # Mock SentenceTransformer
    engine.embedding_model.state_dict.return_value = {
        "layer1.weight": MagicMock(cpu=lambda: MagicMock(numpy=lambda: np.zeros((2, 2)))),
        "layer2.bias": MagicMock(cpu=lambda: MagicMock(numpy=lambda: np.zeros((2,))))
    }
    return engine

def test_build_training_pairs(mock_rag_engine):
    client = EmbeddingFlowerClient(model_path="dummy", rag_engine=mock_rag_engine)
    pairs = client._build_training_pairs()

    # "Kısa döküman" skip edilmeli (len < 50)
    # Diğer ikisi için pair oluşturulmalı
    assert len(pairs) == 2
    assert pairs[0].texts[0] == "Bu birinci cümledir"
    assert "Bu birinci cümledir. Bu da devamıdır." in pairs[0].texts[1]

def test_apply_dp_noise(mock_rag_engine):
    client = EmbeddingFlowerClient(model_path="dummy", rag_engine=mock_rag_engine)
    weights = [np.ones((10, 10)), np.ones(5)]
    global_params = [np.zeros((10, 10)), np.zeros(5)]

    noisy = client._apply_dp_noise(weights, global_params)

    assert len(noisy) == 2
    assert noisy[0].shape == (10, 10)
    assert noisy[1].shape == (5,)
    # Noise eklendiği için değerler değişmiş olmalı (istatistiksel olarak çok düşük ihtimalle aynı kalır)
    assert not np.array_equal(noisy[0], weights[0])

def test_get_parameters(mock_rag_engine):
    client = EmbeddingFlowerClient(model_path="dummy", rag_engine=mock_rag_engine)
    params = client.get_parameters({})

    assert len(params) == 2
    assert params[0].shape == (2, 2)
    assert params[1].shape == (2,)
