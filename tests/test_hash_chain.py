"""Tests for hash-chaining and redaction in the Continuity Bridge."""

from genesis_engine.core.axiom_anchor import AxiomAnchor
from genesis_engine.core.axiomlogix import AxiomLogixTranslator
from genesis_engine.core.continuity_bridge import (
    ContinuityBridge,
    GenesisSoul,
    WisdomEntry,
    redact_sensitive,
)
from genesis_engine.core.deconstruction_engine import DeconstructionEngine


# ---------------------------------------------------------------------------
# Redaction tests
# ---------------------------------------------------------------------------

class TestRedaction:
    def test_redacts_api_key(self):
        text = "My API key is sk-abcdefghij1234567890abcd"
        result = redact_sensitive(text)
        assert "sk-abcdefghij" not in result
        assert "[REDACTED_API_KEY]" in result

    def test_redacts_generic_api_key(self):
        text = "api_key: my_secret_key_12345"
        result = redact_sensitive(text)
        assert "my_secret_key" not in result
        assert "[REDACTED]" in result

    def test_redacts_token(self):
        text = "token = bearer_token_xyz123"
        result = redact_sensitive(text)
        assert "bearer_token" not in result

    def test_redacts_password(self):
        text = "password: supersecret123"
        result = redact_sensitive(text)
        assert "supersecret" not in result

    def test_redacts_email(self):
        text = "Contact me at user@example.com for more info"
        result = redact_sensitive(text)
        assert "user@example.com" not in result
        assert "[REDACTED_EMAIL]" in result

    def test_redacts_phone(self):
        text = "Call me at 555-123-4567"
        result = redact_sensitive(text)
        assert "555-123-4567" not in result
        assert "[REDACTED_PHONE]" in result

    def test_redacts_credit_card(self):
        text = "Card number: 1234-5678-9012-3456"
        result = redact_sensitive(text)
        assert "1234-5678-9012-3456" not in result
        assert "[REDACTED_CC]" in result

    def test_redacts_ssn(self):
        text = "SSN: 123-45-6789"
        result = redact_sensitive(text)
        assert "123-45-6789" not in result
        assert "[REDACTED_SSN]" in result

    def test_preserves_normal_text(self):
        text = "A corporate policy that prioritizes profit over user safety."
        result = redact_sensitive(text)
        assert result == text

    def test_multiple_redactions(self):
        text = "Email: test@test.com, Phone: 555-111-2222, API: sk-1234567890123456789012"
        result = redact_sensitive(text)
        assert "[REDACTED_EMAIL]" in result
        assert "[REDACTED_PHONE]" in result
        assert "[REDACTED_API_KEY]" in result


# ---------------------------------------------------------------------------
# Hash chain tests
# ---------------------------------------------------------------------------

class TestHashChain:
    def test_wisdom_entry_computes_hash(self):
        entry = WisdomEntry(
            source_text="Test text",
            disharmony_summary="Test summary",
            unity_impact=5.0,
            compassion_deficit=3.0,
            resolution_path="reform",
        )
        hash1 = entry.compute_hash("")
        assert len(hash1) == 64  # SHA-256 hex length
        assert hash1.isalnum()

    def test_hash_changes_with_content(self):
        entry1 = WisdomEntry(
            source_text="Text A",
            disharmony_summary="Summary",
            unity_impact=5.0,
            compassion_deficit=3.0,
            resolution_path="reform",
        )
        entry2 = WisdomEntry(
            source_text="Text B",
            disharmony_summary="Summary",
            unity_impact=5.0,
            compassion_deficit=3.0,
            resolution_path="reform",
        )
        hash1 = entry1.compute_hash("")
        hash2 = entry2.compute_hash("")
        assert hash1 != hash2

    def test_hash_chain_includes_prev(self):
        entry = WisdomEntry(
            source_text="Test",
            disharmony_summary="Summary",
            unity_impact=5.0,
            compassion_deficit=3.0,
            resolution_path="reform",
        )
        hash_no_prev = entry.compute_hash("")
        hash_with_prev = entry.compute_hash("abc123")
        assert hash_no_prev != hash_with_prev

    def test_record_wisdom_creates_chain(self):
        anchor = AxiomAnchor()
        translator = AxiomLogixTranslator()
        decon = DeconstructionEngine(anchor=anchor)
        bridge = ContinuityBridge()

        soul = bridge.create_soul(anchor)

        # Add first wisdom entry
        graph1 = translator.translate("Corporation exploits workers.")
        report1 = decon.analyse(graph1)
        soul.record_wisdom(report1, "reform", "Fixed.")

        assert len(soul.wisdom_log) == 1
        entry1 = soul.wisdom_log[0]
        assert entry1.prev_hash == ""
        assert entry1.entry_hash != ""

        # Add second wisdom entry
        graph2 = translator.translate("AI neglects privacy.")
        report2 = decon.analyse(graph2)
        soul.record_wisdom(report2, "reinvention", "Reinvented.")

        assert len(soul.wisdom_log) == 2
        entry2 = soul.wisdom_log[1]
        assert entry2.prev_hash == entry1.entry_hash
        assert entry2.entry_hash != entry1.entry_hash

    def test_verify_chain_valid(self):
        anchor = AxiomAnchor()
        translator = AxiomLogixTranslator()
        decon = DeconstructionEngine(anchor=anchor)
        bridge = ContinuityBridge()

        soul = bridge.create_soul(anchor)

        # Add multiple wisdom entries
        for text in ["Problem 1", "Problem 2", "Problem 3"]:
            graph = translator.translate(f"Corporation {text}")
            report = decon.analyse(graph)
            soul.record_wisdom(report, "reform", f"Resolved {text}")

        is_valid, errors = bridge.verify_wisdom_chain(soul)
        assert is_valid is True
        assert len(errors) == 0

    def test_verify_chain_detects_tamper(self):
        anchor = AxiomAnchor()
        translator = AxiomLogixTranslator()
        decon = DeconstructionEngine(anchor=anchor)
        bridge = ContinuityBridge()

        soul = bridge.create_soul(anchor)

        # Add wisdom entries
        for text in ["Problem 1", "Problem 2"]:
            graph = translator.translate(f"Corporation {text}")
            report = decon.analyse(graph)
            soul.record_wisdom(report, "reform", f"Resolved {text}")

        # Tamper with the first entry's hash
        soul.wisdom_log[0].entry_hash = "tampered_hash"

        is_valid, errors = bridge.verify_wisdom_chain(soul)
        assert is_valid is False
        assert len(errors) >= 1

    def test_verify_chain_detects_broken_link(self):
        anchor = AxiomAnchor()
        translator = AxiomLogixTranslator()
        decon = DeconstructionEngine(anchor=anchor)
        bridge = ContinuityBridge()

        soul = bridge.create_soul(anchor)

        # Add wisdom entries
        for text in ["Problem 1", "Problem 2", "Problem 3"]:
            graph = translator.translate(f"Corporation {text}")
            report = decon.analyse(graph)
            soul.record_wisdom(report, "reform", f"Resolved {text}")

        # Break the chain link
        soul.wisdom_log[1].prev_hash = "wrong_hash"

        is_valid, errors = bridge.verify_wisdom_chain(soul)
        assert is_valid is False
        assert len(errors) >= 1

    def test_empty_chain_is_valid(self):
        bridge = ContinuityBridge()
        soul = bridge.create_soul()

        is_valid, errors = bridge.verify_wisdom_chain(soul)
        assert is_valid is True
        assert len(errors) == 0


# ---------------------------------------------------------------------------
# Redaction in wisdom log tests
# ---------------------------------------------------------------------------

class TestRedactionInWisdom:
    def test_source_text_redacted(self):
        anchor = AxiomAnchor()
        translator = AxiomLogixTranslator()
        decon = DeconstructionEngine(anchor=anchor)
        bridge = ContinuityBridge()

        soul = bridge.create_soul(anchor)

        # Create a report with sensitive data in source text
        graph = translator.translate("Corporation with api_key: secret123 exploits workers")
        report = decon.analyse(graph)
        soul.record_wisdom(report, "reform", "Fixed the issue.")

        entry = soul.wisdom_log[0]
        assert "secret123" not in entry.source_text
        assert "[REDACTED]" in entry.source_text

    def test_resolution_summary_redacted(self):
        anchor = AxiomAnchor()
        translator = AxiomLogixTranslator()
        decon = DeconstructionEngine(anchor=anchor)
        bridge = ContinuityBridge()

        soul = bridge.create_soul(anchor)

        graph = translator.translate("Corporation exploits workers")
        report = decon.analyse(graph)
        soul.record_wisdom(
            report,
            "reform",
            "Contact admin@company.com for details",
        )

        entry = soul.wisdom_log[0]
        assert "admin@company.com" not in entry.resolution_summary
        assert "[REDACTED_EMAIL]" in entry.resolution_summary


# ---------------------------------------------------------------------------
# Export/import with hash chain tests
# ---------------------------------------------------------------------------

class TestExportImportWithChain:
    def test_export_preserves_hashes(self):
        anchor = AxiomAnchor()
        translator = AxiomLogixTranslator()
        decon = DeconstructionEngine(anchor=anchor)
        bridge = ContinuityBridge()

        soul = bridge.create_soul(anchor)

        graph = translator.translate("Corporation exploits workers")
        report = decon.analyse(graph)
        soul.record_wisdom(report, "reform", "Fixed.")

        original_hash = soul.wisdom_log[0].entry_hash

        envelope = bridge.export_soul(soul)
        restored = bridge.import_soul(envelope)

        assert restored is not None
        assert len(restored.wisdom_log) == 1
        assert restored.wisdom_log[0].entry_hash == original_hash

    def test_import_preserves_chain_links(self):
        anchor = AxiomAnchor()
        translator = AxiomLogixTranslator()
        decon = DeconstructionEngine(anchor=anchor)
        bridge = ContinuityBridge()

        soul = bridge.create_soul(anchor)

        for i in range(3):
            graph = translator.translate(f"Problem {i}")
            report = decon.analyse(graph)
            soul.record_wisdom(report, "reform", f"Fixed {i}")

        envelope = bridge.export_soul(soul)
        restored = bridge.import_soul(envelope)

        assert restored is not None
        is_valid, errors = bridge.verify_wisdom_chain(restored)
        assert is_valid is True
