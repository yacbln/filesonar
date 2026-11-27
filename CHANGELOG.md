# Changelog

All notable changes are listed here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[semantic versioning](https://semver.org/).

## [Unreleased]

- Photo descriptions no longer echo the prompt ("not a photo, screenshot, receipt or
  diagram") or state what an image lacks ("there is no visible text or brands"); such
  sentences made unrelated images match searches for those words.
- `filesonar init` creates the config folder with private (0700) permissions.

- `filesonar doctor` warns when Ollama has a model only partly in GPU memory, which
  makes photo descriptions several times slower.

## [0.1.0] - 2026-11-27

First release.

- Index PDFs (text per page, OCR for scanned pages), Word documents, images (vision-model
  captions, OCR, EXIF date and camera) and text files.
- Hybrid search: SQLite FTS5 keywords plus sqlite-vec embeddings, merged with
  reciprocal rank fusion and a relevance cutoff.
- Filters by kind, folder and modification date.
- Incremental indexing with move detection by content hash.
- `filesonar watch` keeps the index current using inotify, with a daily full scan.
- MCP server (`filesonar serve`) with read-only `search_files`, `file_info` and
  `index_status` tools, plus `prepare_file`, which copies an indexed file into an outbox
  folder (OpenClaw's media folder by default) so a chat agent can send it without
  running shell commands. Copies are private and removed after 24 hours.
- Built-in, non-overridable exclusion of keys, credentials and browser data; loopback-only
  model host by default; private index permissions.
- Identical copies of a file (`report (1).pdf`) reuse the existing index entry and are
  shown as one search result with `other_copies`.
- Typing a file name ranks that file first.
- Default exclusions for version control, dependencies, build output and caches apply on
  top of your own patterns; folders where Python packages were installed are skipped.
- Long books and data exports are capped (`max_chunks_per_file`) so they cannot crowd
  out other results; PDF reading stops once that budget is full.
- Robust against real-world files: runaway captions are capped and cleaned, OCR has a
  per-image time limit, truncated images and unusual PNG transparency are handled.
- Files are read once without following symlinks, closing a check-then-read race.
- Google Drive support (opt-in, new): `filesonar drive login` signs in with your own OAuth
  client; Drive files, Google Docs, Sheets and Slides are indexed locally and kept up to
  date through the Drive change feed (`filesonar watch` polls every `poll_minutes`).
  Files that also exist on this computer are matched by MD5 and never downloaded;
  photos are described from Drive's preview image.
- Search results from Drive carry their Drive folder and a private link; `--source`
  / `source` limits results to `local` or `drive`.
- `share_file` MCP tool: a private Drive link for a laptop file, uploading it to a
  `filesonar` folder only if Drive does not already have it.
