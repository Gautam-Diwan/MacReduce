"""Wiki invariants: source preservation, provenance, and navigable synthesis."""

import importlib.util
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "wiki_tool", Path(__file__).resolve().parents[1] / "tools/wiki.py"
)
wiki = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(wiki)


@pytest.fixture
def vault(tmp_path):
    (tmp_path / "raw/sources").mkdir(parents=True)
    (tmp_path / "raw/manifest.json").write_text("[]\n")
    (tmp_path / "wiki").mkdir()
    pages = {
        "index.md": "[Log](log.md)\n[Schema](schema.md)\n",
        "log.md": "Initial entry.\n",
        "schema.md": "Evidence rules.\n",
    }
    for name, body in pages.items():
        (tmp_path / "wiki" / name).write_text(
            wiki.frontmatter(name, "workflow", "active", []) + body
        )
    return tmp_path


def test_intake_preserves_bytes_and_history_and_rejects_duplicate(vault, tmp_path):
    source = tmp_path / "paper.pdf"
    source.write_bytes(b"%PDF-fixture\x00\xff\n")
    old_log = (vault / "wiki/log.md").read_text()
    note = wiki.intake(source, "S01", "Paper", "https://example.org/paper", vault)
    assert (vault / "raw/sources/S01.pdf").read_bytes() == source.read_bytes()
    assert (vault / "wiki/log.md").read_text().startswith(old_log.rstrip())
    assert wiki.metadata(note.read_text())["status"] == "unread"
    assert wiki.lint(vault)["ok"]
    before = (vault / "raw/manifest.json").read_bytes()
    with pytest.raises(ValueError, match="already exists"):
        wiki.intake(source, "S01", "Replacement", "https://example.org/paper", vault)
    assert (vault / "raw/manifest.json").read_bytes() == before
    assert (vault / "raw/sources/S01.pdf").read_bytes() == source.read_bytes()


def test_lint_detects_tampered_snapshot_and_unknown_provenance(vault, tmp_path):
    source = tmp_path / "paper.txt"
    source.write_text("original")
    note = wiki.intake(source, "S01", "Paper", "https://example.org/paper", vault)
    (vault / "raw/sources/S01.txt").write_text("modified")
    note.write_text(note.read_text().replace('sources: ["S01"]', 'sources: ["S99"]'))
    errors = wiki.lint(vault)["errors"]
    assert any("checksum mismatch" in error for error in errors)
    assert any("unknown source ID S99" in error for error in errors)


def test_lint_detects_broken_links_and_missing_index_coverage(vault):
    page = vault / "wiki/unlisted.md"
    page.write_text(wiki.frontmatter("Unlisted", "guide", "active", []) + "[bad](gone.md)")
    errors = wiki.lint(vault)["errors"]
    assert any("broken local link" in error for error in errors)
    assert any("index missing page" in error for error in errors)
    assert any("orphan page" in error for error in errors)


def test_manifest_cannot_reference_outside_snapshot_directory(vault, tmp_path):
    external = tmp_path / "outside.txt"
    external.write_text("outside")
    record = {
        "id": "S01",
        "path": str(external),
        "sha256": wiki.digest(external),
        "bytes": external.stat().st_size,
    }
    (vault / "raw/manifest.json").write_text(json.dumps([record]))
    assert any("outside raw/sources" in error for error in wiki.lint(vault)["errors"])


def test_search_requires_all_words_and_ignores_case(vault):
    (vault / "wiki/schema.md").write_text("ANE optimizer evidence")
    assert wiki.search("ane OPTIMIZER", vault)[0]["path"] == "wiki/schema.md"
    assert not wiki.search("ANE missing", vault)
