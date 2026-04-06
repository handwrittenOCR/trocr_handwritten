import pytest
from pathlib import Path
from unittest.mock import patch

from trocr_handwritten.llm.settings import LLMSettings, OCRSettings
from trocr_handwritten.llm.factory import get_provider, PROVIDERS
from trocr_handwritten.utils.cost_tracker import CostTracker, PRICING
from trocr_handwritten.ner.schemas import (
    ChildInfo,
    PersonInfo,
    BirthActEntity,
    DeathActEntity,
    BIRTH_CSV_COLUMNS,
    DEATH_CSV_COLUMNS,
    flatten_ner_result,
    NERResult,
)


class TestLLMSettings:

    def test_default_provider_is_gemini(self):
        settings = LLMSettings()
        assert settings.provider == "gemini"

    def test_default_model_name(self):
        settings = LLMSettings()
        assert settings.model_name == "gemini-3-pro-preview"

    def test_valid_providers(self):
        for provider in ["openai", "gemini", "mistral"]:
            settings = LLMSettings(provider=provider)
            assert settings.provider == provider

    def test_temperature_default(self):
        settings = LLMSettings()
        assert settings.temperature == 0.0

    def test_max_tokens_default(self):
        settings = LLMSettings()
        assert settings.max_tokens == 4096

    def test_custom_settings(self):
        settings = LLMSettings(
            provider="openai",
            model_name="gpt-4o",
            temperature=0.5,
            max_tokens=2048,
        )
        assert settings.provider == "openai"
        assert settings.model_name == "gpt-4o"
        assert settings.temperature == 0.5
        assert settings.max_tokens == 2048


class TestOCRSettings:

    def test_default_input_dir(self):
        settings = OCRSettings()
        assert settings.input_dir == "data/processed/images"

    def test_default_image_pattern(self):
        settings = OCRSettings()
        assert settings.image_pattern == "*/*/*.jpg"

    def test_default_output_extension(self):
        settings = OCRSettings()
        assert settings.output_extension == ".md"

    def test_nested_llm_settings(self):
        settings = OCRSettings()
        assert isinstance(settings.llm_settings, LLMSettings)


class TestProviderFactory:

    def test_all_providers_registered(self):
        assert "openai" in PROVIDERS
        assert "gemini" in PROVIDERS
        assert "mistral" in PROVIDERS

    def test_get_provider_openai(self):
        from trocr_handwritten.llm.providers.openai import OpenAIProvider

        with patch.object(OpenAIProvider, "_init_client"):
            settings = LLMSettings(provider="openai")
            provider = get_provider(settings)
            assert isinstance(provider, OpenAIProvider)

    def test_get_provider_gemini(self):
        from trocr_handwritten.llm.providers.gemini import GeminiProvider

        with patch.object(GeminiProvider, "_init_client"):
            settings = LLMSettings(provider="gemini")
            provider = get_provider(settings)
            assert isinstance(provider, GeminiProvider)

    def test_get_provider_mistral(self):
        from trocr_handwritten.llm.providers.mistral import MistralProvider

        with patch.object(MistralProvider, "_init_client"):
            settings = LLMSettings(provider="mistral")
            provider = get_provider(settings)
            assert isinstance(provider, MistralProvider)


class TestCostTracker:

    def test_initial_values(self):
        tracker = CostTracker(model_name="gpt-4o")
        assert tracker.input_tokens == 0
        assert tracker.output_tokens == 0
        assert tracker.total_calls == 0

    def test_add_usage(self):
        tracker = CostTracker(model_name="gpt-4o")
        tracker.add_usage(1000, 500)
        assert tracker.input_tokens == 1000
        assert tracker.output_tokens == 500
        assert tracker.total_calls == 1

    def test_add_usage_accumulates(self):
        tracker = CostTracker(model_name="gpt-4o")
        tracker.add_usage(1000, 500)
        tracker.add_usage(2000, 1000)
        assert tracker.input_tokens == 3000
        assert tracker.output_tokens == 1500
        assert tracker.total_calls == 2

    def test_get_cost_known_model(self):
        tracker = CostTracker(model_name="gemini-2.0-flash")
        tracker.add_usage(1_000_000, 1_000_000)
        costs = tracker.get_cost()
        expected = 0.085 + 0.339
        assert costs["total"] == pytest.approx(expected)

    def test_get_cost_unknown_model(self):
        tracker = CostTracker(model_name="unknown-model")
        tracker.add_usage(1_000_000, 1_000_000)
        costs = tracker.get_cost()
        assert costs["total"] == 0.0

    def test_get_cost_with_thinking_tokens(self):
        tracker = CostTracker(model_name="gemini-3.1-pro-preview")
        tracker.add_usage(1_000_000, 500_000, thinking_tokens=200_000)
        costs = tracker.get_cost()
        # Thinking tokens billed at output rate
        assert costs["thinking_cost"] == pytest.approx(200_000 / 1_000_000 * 10.178)
        assert costs["total"] == pytest.approx(
            costs["input_cost"] + costs["output_cost"] + costs["thinking_cost"]
        )

    def test_summary_format(self):
        tracker = CostTracker(model_name="gpt-4o")
        tracker.add_usage(1000, 500)
        summary = tracker.summary()
        assert "gpt-4o" in summary
        assert "1,000" in summary
        assert "500" in summary
        assert "EUR" in summary

    def test_pricing_contains_expected_models(self):
        expected_models = [
            "gpt-4o",
            "gpt-5.2",
            "gemini-2.0-flash",
            "gemini-3-pro-preview",
            "mistral-large-latest",
            "pixtral-large-latest",
        ]
        for model in expected_models:
            assert model in PRICING
            assert "input" in PRICING[model]
            assert "output" in PRICING[model]


class TestNERSchemas:

    def test_child_info_has_qualifier(self):
        child = ChildInfo(name="Victoire", sex="femme", qualifier="négritte")
        assert child.qualifier == "négritte"

    def test_child_info_no_age_or_occupation(self):
        fields = ChildInfo.model_fields
        assert "age" not in fields
        assert "occupation" not in fields

    def test_person_info_has_qualifier(self):
        person = PersonInfo(name="Thomas", sex="homme", qualifier="nègre", age="40")
        assert person.qualifier == "nègre"

    def test_qualifier_defaults_to_none(self):
        assert ChildInfo().qualifier is None
        assert PersonInfo().qualifier is None

    def test_birth_csv_columns_has_child_qualifier(self):
        assert "child_qualifier" in BIRTH_CSV_COLUMNS

    def test_death_csv_columns_has_person_qualifier(self):
        assert "person_qualifier" in DEATH_CSV_COLUMNS

    def test_birth_csv_columns_no_child_age(self):
        assert "child_age" not in BIRTH_CSV_COLUMNS

    def test_owner_fields_in_birth_columns(self):
        assert "owner_commune" in BIRTH_CSV_COLUMNS
        assert "owner_residence" in BIRTH_CSV_COLUMNS

    def test_owner_fields_in_death_columns(self):
        assert "owner_commune" in DEATH_CSV_COLUMNS
        assert "owner_residence" in DEATH_CSV_COLUMNS

    def test_flatten_birth_includes_qualifier(self):
        entity = BirthActEntity(
            child=ChildInfo(name="Rose", sex="femme", qualifier="négrette"),
            owner_commune="Marin",
            owner_residence="habitation Beauséjour",
        )
        result = NERResult(
            act_id="test_1",
            act_type="naissance",
            extraction_method="regex",
            birth_act=entity,
            raw_marge="",
            raw_plein_texte="",
        )
        row = flatten_ner_result(result)
        assert row["child_qualifier"] == "négrette"
        assert row["owner_commune"] == "Marin"
        assert row["owner_residence"] == "habitation Beauséjour"

    def test_flatten_death_includes_qualifier(self):
        entity = DeathActEntity(
            person=PersonInfo(name="Jules", sex="homme", qualifier="mulâtre", age="30"),
            owner_commune="Abymes",
        )
        result = NERResult(
            act_id="test_2",
            act_type="deces",
            extraction_method="regex",
            death_act=entity,
            raw_marge="",
            raw_plein_texte="",
        )
        row = flatten_ner_result(result)
        assert row["person_qualifier"] == "mulâtre"
        assert row["owner_commune"] == "Abymes"


class TestOpenAIProvider:

    def test_is_new_model_gpt5(self):
        from trocr_handwritten.llm.providers.openai import OpenAIProvider

        with patch.object(OpenAIProvider, "_init_client"):
            settings = LLMSettings(provider="openai", model_name="gpt-5.2")
            provider = OpenAIProvider(settings)
            assert provider._is_new_model("gpt-5.2") is True

    def test_is_new_model_gpt4(self):
        from trocr_handwritten.llm.providers.openai import OpenAIProvider

        with patch.object(OpenAIProvider, "_init_client"):
            settings = LLMSettings(provider="openai", model_name="gpt-4o")
            provider = OpenAIProvider(settings)
            assert provider._is_new_model("gpt-4o") is False

    def test_is_new_model_o1(self):
        from trocr_handwritten.llm.providers.openai import OpenAIProvider

        with patch.object(OpenAIProvider, "_init_client"):
            settings = LLMSettings(provider="openai", model_name="o1-preview")
            provider = OpenAIProvider(settings)
            assert provider._is_new_model("o1-preview") is True


class TestProviderBase:

    def test_encode_image_base64(self, tmp_path):
        from trocr_handwritten.llm.providers.gemini import GeminiProvider

        image_path = tmp_path / "test.jpg"
        image_path.write_bytes(b"fake image content")

        with patch.object(GeminiProvider, "_init_client"):
            settings = LLMSettings(provider="gemini")
            provider = GeminiProvider(settings)
            encoded = provider._encode_image_base64(image_path)

        assert isinstance(encoded, str)
        assert len(encoded) > 0

    def test_get_mime_type(self):
        from trocr_handwritten.llm.providers.gemini import GeminiProvider

        with patch.object(GeminiProvider, "_init_client"):
            settings = LLMSettings(provider="gemini")
            provider = GeminiProvider(settings)

        assert provider._get_mime_type(Path("test.jpg")) == "image/jpeg"
        assert provider._get_mime_type(Path("test.png")) == "image/png"
        assert provider._get_mime_type(Path("test.webp")) == "image/webp"
