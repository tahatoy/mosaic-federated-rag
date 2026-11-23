import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "ai"))
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from backend.ai_engine import strip_pii_from_chunks, detect_branch_from_query

def test_strip_pii_from_chunks():
    chunks = [
        {
            "content": "Kullanıcı Hakan Serter'in 12345678901 nolu hesabı.",
            "metadata": {
                "musteri_adi": "Hakan Serter",
                "tc_no": "12345678901",
                "source": "branch_02_doc"
            },
            "branch": "branch_02"
        }
    ]

    cleaned = strip_pii_from_chunks(chunks)

    assert "musteri_adi" not in cleaned[0]["metadata"]
    assert "tc_no" not in cleaned[0]["metadata"]
    assert cleaned[0]["metadata"]["source"] == "branch_02_doc"
    assert "Hakan Serter" in cleaned[0]["content"]  # Content stays same

def test_detect_branch_from_query():
    # Test Branch 1
    bid1, detected1 = detect_branch_from_query("Şube 1'in verileri neler?")
    assert bid1 == 1
    assert detected1 is True

    # Test Branch 2
    bid2, detected2 = detect_branch_from_query("ikinci şube dökümanları")
    assert bid2 == 2
    assert detected2 is True

    # Test Global
    bid0, detected0 = detect_branch_from_query("genel finansal durum")
    assert bid0 == 0
    assert detected0 is False

def test_detect_branch_from_query_variation():
    bid, detected = detect_branch_from_query("branch 2'de ne var?")
    assert bid == 2
    assert detected is True
