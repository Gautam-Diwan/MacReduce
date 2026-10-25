#!/usr/bin/env python3
"""Local wiki intake, search, and structural checks; no LLM or network calls."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
STATUSES = {"active", "proposed", "measured", "unread", "historical", "template"}
LINK = re.compile(r"!?\[[^\]]*\]\((<[^>]+>|[^\s)]+)(?:\s+\"[^\"]*\")?\)")


def metadata(text: str) -> dict:
    if not text.startswith("---\n") or "\n---\n" not in text[4:]:
        raise ValueError("missing frontmatter")
    block = text.split("\n---\n", 1)[0][4:]
    result = {}
    for line in block.splitlines():
        key, sep, value = line.partition(":")
        if not sep or not key.strip():
            raise ValueError("frontmatter requires key: value fields")
        if key in result:
            raise ValueError(f"duplicate metadata field: {key}")
        value = value.strip()
        result[key] = json.loads(value) if value.startswith(('"', '[')) else value
    return result


def frontmatter(title: str, kind: str, status: str, sources: list[str]) -> str:
    date = datetime.now(timezone.utc).date().isoformat()
    return (
        f"---\ntitle: {json.dumps(title, ensure_ascii=False)}\ntype: {kind}\n"
        f"status: {status}\nupdated: {date}\nsources: {json.dumps(sources)}\n---\n\n"
    )


def inside(root: Path, path: Path) -> bool:
    return path.resolve().is_relative_to(root.resolve())


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def lint(root: Path = ROOT) -> dict:
    root = root.resolve()
    errors, warnings = [], []
    try:
        manifest = json.loads((root / "raw/manifest.json").read_text())
        if not isinstance(manifest, list):
            raise ValueError("manifest must be a list")
    except (OSError, ValueError) as exc:
        return {"ok": False, "errors": [f"manifest: {exc}"], "warnings": []}
    ids = set()
    for record in manifest:
        if not isinstance(record, dict):
            errors.append("manifest record must be an object")
            continue
        ident = record.get("id")
        if not isinstance(ident, str) or not re.fullmatch(r"S\d+", ident):
            errors.append(f"invalid source ID: {ident!r}")
            continue
        if ident in ids:
            errors.append(f"duplicate source ID: {ident}")
        ids.add(ident)
        try:
            path = root / record["path"]
            if not inside(root / "raw/sources", path) or not path.is_file():
                raise ValueError("snapshot missing or outside raw/sources")
            if digest(path) != record["sha256"]:
                errors.append(f"{ident}: snapshot checksum mismatch")
            if path.stat().st_size != record["bytes"]:
                errors.append(f"{ident}: snapshot byte count mismatch")
            for key in ("title", "url", "scope", "retrieved_at"):
                if not isinstance(record.get(key), str) or not record[key]:
                    errors.append(f"{ident}: missing {key}")
        except (KeyError, OSError, TypeError, ValueError) as exc:
            errors.append(f"{ident}: {exc}")
        if not (root / f"wiki/sources/{ident}.md").is_file():
            errors.append(f"{ident}: missing source note")
    pages = sorted((root / "wiki").rglob("*.md"))
    links = {page.resolve(): set() for page in pages}
    for page in pages:
        label = str(page.relative_to(root))
        text = page.read_text()
        try:
            meta = metadata(text)
            for key in ("title", "type", "status", "updated", "sources"):
                if key not in meta or meta[key] == "":
                    errors.append(f"{label}: missing {key}")
            if meta.get("status") not in STATUSES:
                errors.append(f"{label}: invalid status")
            datetime.strptime(str(meta.get("updated")), "%Y-%m-%d")
            sources = meta.get("sources")
            if not isinstance(sources, list) or any(not isinstance(s, str) for s in sources):
                raise ValueError("sources must be an inline JSON array of IDs")
            for ident in sources:
                if ident not in ids:
                    errors.append(f"{label}: unknown source ID {ident}")
            if meta.get("status") == "unread":
                warnings.append(f"{label}: source synthesis pending")
        except (ValueError, TypeError) as exc:
            errors.append(f"{label}: {exc}")
        # Fenced examples can contain placeholder paths; check rendered prose links.
        prose = re.sub(r"(?ms)^```[^\n]*\n.*?^```\s*$", "", text)
        for match in LINK.finditer(prose):
            raw = match.group(1).strip("<>")
            parsed = urlsplit(raw)
            if parsed.scheme or parsed.netloc or not parsed.path:
                continue
            target = (page.parent / unquote(parsed.path)).resolve()
            if not inside(root, target):
                errors.append(f"{label}: link escapes project: {raw}")
            elif not target.exists():
                errors.append(f"{label}: broken local link: {raw}")
            elif target in links:
                links[page.resolve()].add(target)
    index = (root / "wiki/index.md").resolve()
    if index not in links:
        errors.append("wiki/index.md missing")
    else:
        incoming = set().union(*links.values()) if links else set()
        for page in links:
            if page != index and page not in links[index]:
                errors.append(f"index missing page: {page.relative_to(root)}")
            if page != index and page not in incoming:
                errors.append(f"orphan page: {page.relative_to(root)}")
    return {"ok": not errors, "pages": len(pages), "sources": len(ids),
            "errors": errors, "warnings": warnings}


def search(query: str, root: Path = ROOT) -> list[dict]:
    terms = query.casefold().split()
    if not terms:
        raise ValueError("search query must contain text")
    results = []
    for page in sorted((root / "wiki").rglob("*.md")):
        text = page.read_text()
        if all(term in text.casefold() for term in terms):
            excerpts = [f"{i}: {line.strip()}" for i, line in enumerate(text.splitlines(), 1)
                        if any(term in line.casefold() for term in terms)]
            results.append({"path": str(page.relative_to(root)), "matches": excerpts[:4]})
    return results


def intake(source: Path, ident: str, title: str, url: str, root: Path = ROOT) -> Path:
    """Append one local file. Call serially; concurrent writers are unsupported."""
    root = root.resolve()
    if not re.fullmatch(r"S\d+", ident):
        raise ValueError("ID must be S followed by digits")
    if not title.strip() or urlsplit(url).scheme not in {"https", "http"}:
        raise ValueError("a title and an HTTP(S) original URL are required")
    if not source.is_file():
        raise ValueError("source must be an existing local file")
    manifest_path = root / "raw/manifest.json"
    records = json.loads(manifest_path.read_text())
    if any(record["id"] == ident for record in records):
        raise ValueError(f"{ident} already exists; revisions require a new ID")
    suffix = source.suffix.lower()
    if not re.fullmatch(r"\.[a-z0-9]{1,10}", suffix):
        suffix = ".bin"
    destination = root / "raw/sources" / f"{ident}{suffix}"
    note = root / "wiki/sources" / f"{ident}.md"
    index, log = root / "wiki/index.md", root / "wiki/log.md"
    for path in (destination, note):
        if path.exists():
            raise ValueError(f"refusing to overwrite {path}")
    # Validate required existing files before mutating anything.
    index_text, log_text = index.read_text(), log.read_text()
    data = source.read_bytes()
    record = {"id": ident, "title": title.strip(), "url": url,
              "scope": "unread; intake only", "retrieved_at": datetime.now(timezone.utc).isoformat(),
              "mutable_url": True, "path": str(destination.relative_to(root)),
              "status": "retrieved", "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
    note_text = frontmatter(f"{ident} — {title.strip()}", "source", "unread", [ident])
    note_text += (
        f"# {ident} — {title.strip()}\n\n"
        f"**Original URL:** <{url}>\n\n"
        f"**Snapshot:** [saved bytes](../../{record['path']})\n\n"
        f"**Retrieved:** {record['retrieved_at']}  \n"
        f"**SHA-256:** `{record['sha256']}`\n\n"
        "**Reading scope:** unread. Intake copied the source; it did not evaluate it.\n\n"
        "## Findings and limitations\n\nPending source reading and synthesis.\n\n"
        "## Affected topics\n\nFollow [the schema](../schema.md) and update related pages.\n"
    )
    # A local append operation, not a database transaction. Lint detects partial writes.
    # Existing source snapshots are never opened for writing.
    destination.parent.mkdir(parents=True, exist_ok=True)
    note.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as stream:
        stream.write(data)
    with note.open("x") as stream:
        stream.write(note_text)
    temporary = manifest_path.with_suffix(".json.tmp")
    with temporary.open("x") as stream:
        stream.write(json.dumps(records + [record], indent=2) + "\n")
    temporary.replace(manifest_path)
    safe_title = title.strip().replace("[", "(").replace("]", ")").replace("\n", " ")
    index.write_text(index_text.rstrip() + f"\n\n- [{ident} — {safe_title}](sources/{ident}.md)\n")
    log.write_text(log_text.rstrip() + f"\n\n## {record['retrieved_at']} — Intake {ident}\n\n"
                   f"Added [{ident}](sources/{ident}.md) as unread; source synthesis is pending.\n")
    return note


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("lint", help="check metadata, snapshots, links, and index coverage")
    find = sub.add_parser("search", help="search wiki text; all query words must occur")
    find.add_argument("query")
    add = sub.add_parser("intake", help="copy a local source and create an unread source note")
    add.add_argument("source", type=Path)
    for flag in ("id", "title", "url"):
        add.add_argument(f"--{flag}", required=True)
    args = parser.parse_args()
    try:
        if args.command == "lint":
            report = lint()
            print(json.dumps(report, indent=2))
            return 0 if report["ok"] else 1
        if args.command == "search":
            print(json.dumps(search(args.query), indent=2, ensure_ascii=False))
        else:
            print(intake(args.source, args.id, args.title, args.url))
        return 0
    except (OSError, ValueError, KeyError) as exc:
        print(f"wiki: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
