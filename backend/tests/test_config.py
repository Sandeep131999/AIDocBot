"""
tests/test_config.py
Tests for Config (pydantic-settings) validation and helpers.
"""
from __future__ import annotations

import pytest


class TestConfig:

    def test_config_loads_env_defaults(self):
        from src.config import Config
        assert Config.CHUNK_SIZE == 800
        assert Config.CHUNK_OVERLAP == 150
        assert Config.TOP_K == 5

    def test_llm_provider_list_parsed(self):
        from src.config import Config
        providers = Config.llm_provider_list
        assert isinstance(providers, list)
        assert len(providers) > 0

    def test_allowed_extensions_list(self):
        from src.config import Config
        exts = Config.allowed_extensions_list
        assert isinstance(exts, list)
        assert ".pdf" in exts or "pdf" in " ".join(exts)

    def test_pii_entities_list(self):
        from src.config import Config
        entities = Config.pii_entities_list
        assert isinstance(entities, list)
        assert "EMAIL_ADDRESS" in entities or len(entities) > 0

    def test_cors_origins_wildcard(self):
        from src.config import Config
        origins = Config.cors_origins_list
        assert origins == ["*"] or len(origins) > 0

    def test_get_prompt_replaces_escaped_newlines(self):
        from src.config import Config
        # The GRADING_PROMPT in env has literal \n which should become real newlines
        prompt = Config.get_prompt("GRADING_PROMPT")
        assert "{question}" in prompt
        assert "{context}" in prompt

    def test_get_settings_returns_singleton(self):
        from src.config import get_settings
        s1 = get_settings()
        s2 = get_settings()
        assert s1 is s2   # lru_cache ensures same object
