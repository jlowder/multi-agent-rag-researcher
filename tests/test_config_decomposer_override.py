"""Unit tests for the DECOMPOSER_MODEL per-agent override chain
(env var -> Config field -> get_agent_config -> run_model model lookup).

No live LLM calls: run_model is not involved; only config resolution is
exercised. Always resets the global config so no stale state leaks into
other tests.
"""

import utils.config as config_mod
from utils.config import get_config


class TestDecomposerModelOverride:
    def setup_method(self):
        config_mod.reset_config()

    def teardown_method(self):
        config_mod.reset_config()

    def test_decomposer_model_env_overrides_global(self, monkeypatch):
        monkeypatch.setenv("LLM_MODEL", "Ornith-1.0-35B-MLX-oQ8")
        monkeypatch.setenv(
            "DECOMPOSER_MODEL", "Qwen3-Coder-Next-MLX-6bit"
        )
        cfg = get_config()
        assert cfg.get_agent_config("decomposer").model == (
            "Qwen3-Coder-Next-MLX-6bit"
        )
        # Every other agent stays on the global default.
        assert cfg.get_agent_config("writer").model == "Ornith-1.0-35B-MLX-oQ8"
        assert cfg.get_agent_config("verifier").model == (
            "Ornith-1.0-35B-MLX-oQ8"
        )
        assert cfg.get_agent_config("sufficiency").model == (
            "Ornith-1.0-35B-MLX-oQ8"
        )

    def test_decomposer_inherits_global_when_env_absent(self, monkeypatch):
        monkeypatch.setenv("LLM_MODEL", "Ornith-1.0-35B-MLX-oQ8")
        monkeypatch.delenv("DECOMPOSER_MODEL", raising=False)
        cfg = get_config()
        assert cfg.get_agent_config("decomposer").model == (
            "Ornith-1.0-35B-MLX-oQ8"
        )

    def test_decomposer_endpoint_and_key_stay_global(self, monkeypatch):
        # The override is model-only: endpoint/api_key must not change.
        monkeypatch.setenv("LLM_ENDPOINT", "http://localhost:8080/v1")
        monkeypatch.setenv("LLM_API_KEY", "k")
        monkeypatch.setenv("DECOMPOSER_MODEL", "Qwen3-Coder-Next-MLX-6bit")
        cfg = get_config()
        dec = cfg.get_agent_config("decomposer")
        assert dec.endpoint == "http://localhost:8080/v1"
        assert dec.api_key == "k"
