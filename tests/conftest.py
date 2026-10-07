import pytest

# Engine settings a developer's .env may change; tests always start from the code defaults.
ENGINE_SETTINGS = ("MAX_AGENT_STEPS", "LLM_TIMEOUT_SECONDS", "RUN_TIMEOUT_S", "GROQ_REASONING_EFFORT")


@pytest.fixture(autouse=True)
def never_call_real_llms(request, monkeypatch):
    """Tests run in fake mode with default settings, even if .env says otherwise
    (app.main loads .env on import). Only tests marked @pytest.mark.live may use real providers.
    """
    if request.node.get_closest_marker("live") is None:
        monkeypatch.setenv("LLM_MODE", "fake")
        for name in ENGINE_SETTINGS:
            monkeypatch.delenv(name, raising=False)
