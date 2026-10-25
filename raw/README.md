# Source snapshots

`manifest.json` records the URL, retrieval time in UTC, inspected scope, byte length, and SHA-256 of each source snapshot. Files in `sources/` are immutable reference material; new revisions get new IDs. Source cards under `wiki/sources/` are editable synthesis and identify what was actually read.

Checksums identify saved bytes, not Git commits. Branch, documentation, and unversioned paper URLs may change regardless of the original `mutable_url` hint. Refresh before relying on current APIs. Source copyrights and licenses remain with their owners; local research storage is not permission for redistribution. No downloaded content is executed by wiki tools.

Use `python tools/wiki.py intake` to append a local document and then explicitly ask an assistant to read and synthesize it. Run `python tools/wiki.py lint` to verify hashes and local references. Lint is structural, not factual verification.
