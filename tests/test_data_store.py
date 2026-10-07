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


# --- Retrieval: the live-demo scenario must be found by plain keyword search ---

RESIDENCY_QUERIES = [
    "data residency",
    "where is data stored",
    "Flag any conflict between this vendor's data terms and our data residency policy",
    "Canadian and EU customer data location",
]


@pytest.mark.parametrize("query", RESIDENCY_QUERIES)
def test_residency_queries_find_fjord_clause(query):
    top = data_store.bm25_search(query, data_store.request_chunks("3"), k=3)
    assert any(KEY_QUOTES["fjord-dpa"] in c.text for c in top), [c.chunk_id for c in top]


@pytest.mark.parametrize("query", RESIDENCY_QUERIES)
def test_residency_queries_find_residency_policy(query):
    top = data_store.bm25_search(query, data_store.policy_chunks(), k=3)
    assert "POL-03" in [c.chunk_id for c in top]


def test_bm25_search_empty_corpus():
    assert data_store.bm25_search("anything", []) == []


# --- Cached data must not be shared between callers ---


def test_mutating_returned_data_does_not_leak():
    data_store.get_request("1")["amount"] = 0
    data_store.list_requests()[0]["documents"].clear()
    data_store.list_vendors()[0]["name"] = "Hacked"
    data_store.get_vendor("V-1001")["status"] = "blocked"
    data_store.get_contract("V-1001")["max_annual_increase_pct"] = 99
    data_store.get_purchase_history("V-1001")[0]["amount"] = -1
    data_store.get_policies()[0]["text"] = "changed"
    data_store.get_documents("1")[0]["text"] = "changed"
    data_store.load_presets()[0].tools.append("api_data")

    assert data_store.get_request("1")["amount"] == 64900
    assert data_store.list_requests()[0]["documents"] == ["northbeam-master-agreement", "northbeam-renewal-order-form"]
    assert data_store.list_vendors()[0]["name"] == "Northbeam Analytics"
    assert data_store.get_vendor("V-1001")["status"] == "active"
    assert data_store.get_contract("V-1001")["max_annual_increase_pct"] == 7
    assert data_store.get_purchase_history("V-1001")[0]["amount"] == 51500
    assert data_store.get_policies()[0]["text"] != "changed"
    assert data_store.get_documents("1")[0]["text"] != "changed"
    assert data_store.load_presets()[0].tools == ["api_data", "document_retrieval", "company_context"]


# --- Review follow-ups ---


def test_list_vendors_has_name_domain_status():
    for vendor in data_store.list_vendors():
        assert vendor["name"]
        assert vendor["email_domain"] == vendor["contact_email"].split("@")[1]
        assert vendor["status"] in {"active", "onboarding"}


def test_presets_are_exactly_the_two_and_never_use_fjord():
    presets = data_store.load_presets()
    assert sorted(p.name for p in presets) == ["Duplicate Vendor Check", "Renewal Check"]
    assert "3" not in {p.default_request_id for p in presets}  # Fjord is built live on stage


def test_vendor_data_is_obviously_fictional():
    for vendor in data_store.list_vendors():
        assert "(fictional)" in vendor["bank_account"]
        assert vendor["email_domain"].endswith(".example")  # reserved domain, can't be real


def test_bm25_returns_top_k_even_when_no_score_is_positive():
    # Tiny corpus: the only query word appears in every chunk, so its IDF is
    # negative, and an unknown word scores 0. Results must still come back.
    chunks = [
        data_store.Chunk(doc_name="d", chunk_id=f"d#{i}", text=text)
        for i, text in enumerate(["vendor alpha", "vendor beta", "vendor gamma"], start=1)
    ]
    bm25 = data_store.BM25Okapi([data_store.tokenize(c.text) for c in chunks])
    assert all(score <= 0 for score in bm25.get_scores(["vendor"]))

    assert len(data_store.bm25_search("vendor", chunks, k=2)) == 2
    # All scores equal (zero): ties keep the original order.
    assert [c.chunk_id for c in data_store.bm25_search("unmatched", chunks, k=3)] == ["d#1", "d#2", "d#3"]


def test_chunker_keeps_heading_with_its_paragraph():
    for request in data_store.list_requests():
        for doc in data_store.get_documents(request["id"]):
            for chunk in data_store.chunk_document(doc["name"], doc["text"]):
                last_block = chunk.text.split("\n\n")[-1]
                assert not last_block.startswith("#"), f"{chunk.chunk_id} ends with a heading"
