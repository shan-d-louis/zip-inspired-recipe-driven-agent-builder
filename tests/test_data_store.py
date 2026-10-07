import pytest

from app import data_store
from app.models import OutputFormat

# The exact passages later citations will quote. Each must exist verbatim
# in its document and must never be split across chunks.
KEY_QUOTES = {
    "northbeam-master-agreement": "annual fee increases shall not exceed seven percent (7%)",
    "northbeam-renewal-order-form": "automatically renew for successive twelve (12) month terms",
    "northbeam-inc-vendor-application": "formerly operating as Northbeam Analytics",
    "fjord-dpa": "Customer Data will be stored and processed exclusively in data centers located in the United States",
    "quillstone-msa": "Consultant's liability under this Agreement shall be unlimited",
}


def test_four_requests_with_vendors_and_documents():
    requests = data_store.list_requests()
    assert [r["id"] for r in requests] == ["1", "2", "3", "4"]
    for request in requests:
        assert data_store.get_vendor(request["vendor_id"]) is not None
        docs = data_store.get_documents(request["id"])
        assert 1 <= len(docs) <= 3
        assert [d["name"] for d in docs] == request["documents"]


@pytest.mark.parametrize("request_id", ["1", "2", "3", "4"])
def test_documents_are_200_to_500_words(request_id):
    for doc in data_store.get_documents(request_id):
        words = len(doc["text"].split())
        assert 200 <= words <= 500, f"{doc['name']} has {words} words"


def test_unknown_request():
    assert data_store.get_request("99") is None
    assert data_store.get_documents("99") == []


def test_duplicate_vendor_scenario_data():
    new_vendor = data_store.get_vendor(data_store.get_request("2")["vendor_id"])
    active_domains = {
        v["contact_email"].split("@")[1] for v in data_store.list_vendors() if v["status"] == "active"
    }
    assert new_vendor["contact_email"].split("@")[1] in active_domains


def test_renewal_scenario_data():
    contract = data_store.get_contract("V-1001")
    assert contract["max_annual_increase_pct"] == 7
    renewal_amount = data_store.get_request("1")["amount"]
    assert round((renewal_amount / contract["annual_fee"] - 1) * 100) == 18
    assert len(data_store.get_purchase_history("V-1001")) == 3


def test_eight_policies():
    policies = data_store.get_policies()
    assert [p["id"] for p in policies] == [f"POL-0{i}" for i in range(1, 9)]
    assert all(p["title"] and p["text"] for p in policies)


def test_presets():
    presets = {p.id: p for p in data_store.load_presets()}
    assert set(presets) == {"renewal-check", "duplicate-vendor-check"}
    assert presets["renewal-check"].output_format is OutputFormat.MARKDOWN
    assert presets["renewal-check"].default_request_id == "1"
    assert presets["duplicate-vendor-check"].output_format is OutputFormat.STRUCTURED
    assert presets["duplicate-vendor-check"].default_request_id == "2"
    assert all(p.is_preset for p in presets.values())


def test_chunker_keeps_all_text_and_numbers_chunks():
    doc = data_store.get_documents("3")[0]
    chunks = data_store.chunk_document(doc["name"], doc["text"])
    assert len(chunks) > 1
    assert [c.chunk_id for c in chunks] == [f"fjord-dpa#{i}" for i in range(1, len(chunks) + 1)]
    assert " ".join(c.text for c in chunks).split() == doc["text"].split()


@pytest.mark.parametrize("doc_name,quote", KEY_QUOTES.items())
def test_key_quotes_each_inside_one_chunk(doc_name, quote):
    text = data_store._read_document(doc_name)
    chunks = data_store.chunk_document(doc_name, text)
    assert sum(quote in c.text for c in chunks) == 1
