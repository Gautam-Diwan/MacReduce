---
title: "Using and maintaining the MacReduce wiki"
type: workflow
status: active
updated: 2026-10-23
sources: ["S01"]
---

# Using and maintaining the MacReduce wiki

This is a local, agent-maintained research wiki following Karpathy's pattern. Read [the index](index.md) to enter by topic; use [the log](log.md) to see changes over time. Open this folder in a Markdown editor or open the project as an Obsidian vault. No hosted service or model API key is required. A coding assistant performs the synthesis; the included tool only searches, scaffolds source intake, and checks consistency.

## Three layers

- `raw/` stores source snapshots and a SHA-256 manifest. Existing snapshots are immutable. A newer source revision gets a new record.
- `wiki/` contains maintained interpretations, implementation guides, experiments, and decisions. These pages are editable and can be corrected.
- `AGENTS.md` and [schema.md](schema.md) define how a future assistant should ingest, answer, and maintain the collection.

## Start a session

Tell your assistant: “Read AGENTS.md and wiki/index.md. Summarize the current evidence and next unfinished milestone before making changes.” Ask it to read linked source notes and primary snapshots for factual claims, rather than recursively trusting earlier summaries.

## Ingest a source

1. Save a paper or article locally, record its original URL, and retain its license. Do not treat an LLM answer as a primary source.
2. Run `python tools/wiki.py intake /path/to/source.pdf --id S23 --title 'Paper title' --url 'https://original.example/paper'`. This copies the bytes, records a checksum, creates an unread source note, and updates the index and log. Use the next unused ID. Intake does not summarize or evaluate the document. Run one intake at a time: this local helper is not a transactional multi-user database. If an operation is interrupted, run lint and reconcile the manifest, new note, index, and log before retrying; preserve any already copied source bytes.
3. Ask: “Ingest S23 using wiki/schema.md. Read the source, summarize the actual method and limitations, compare it with our current claims, and update every affected page. Distinguish what you inspected from what remains unread.” For PDFs the assistant must use a PDF reader; this CLI does not extract text.
4. Review the source note, especially the reading scope and unsupported claims. Resolve conflicts explicitly in [the gaps ledger](research/gaps.md).
5. Run `python tools/wiki.py lint`; append a human or agent review outcome to the log. A green structural check is not a factual review.

## Query and retain useful answers

Use `python tools/wiki.py search 'ANE optimizer'` or `rg -n 'ANE' wiki`. Ask the assistant to answer with local source links and explicit evidence labels. Substantial new answers can become pages under `queries/`; link them from the index and related topics and append a log entry. Keep speculative answers labeled hypotheses.

## Periodic maintenance

After a code change, update [project state](project/state.md), the affected guide, and the change log. After an experiment, create a report from [the experiment template](templates/experiment.md) and attach raw measurements. Before a proposal or novelty claim, refresh the related-work search and read full papers for the closest competitors. Do not change a conclusion merely because a newer summary says something different.

The process is deliberately reviewable. Ingest, query, and lint are explicit operations, not a background service that silently rewrites the project.
