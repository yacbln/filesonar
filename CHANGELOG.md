# Changelog

All notable changes are listed here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[semantic versioning](https://semver.org/).

## [Unreleased]

## [0.2.0] - 2026-11-27

- New formats: PowerPoint and LibreOffice Impress slides (`.pptx`, `.odp`; results point
  to the slide, speaker notes included), Excel and LibreOffice Calc sheets (`.xlsx`,
  `.ods`), LibreOffice Writer (`.odt`) and e-books (`.epub`, chapters in reading
  order). They are read with the standard library and `defusedxml`; an archive may not
  unpack to more than 200 MB, and XML with entity declarations is refused.
- Web pages (`.html`) are indexed as the text a reader sees, without markup, scripts or
  styles. Pages indexed by an earlier version are read again once.
- Google Slides are read per slide (results point to the slide) and Google Sheets with
  every sheet, by exporting them as `.pptx` / `.xlsx`; CSV only ever had the first
  sheet. A deck too large for Google's 10 MB export limit falls back to plain text.
  Slides and Sheets already indexed are read again once, on the next full Drive scan.
- Placeholder titles and authors ("Unknown Title", "Untitled") are not indexed.
- `scripts/eval_search.py --same-files-as INDEX` measures two indexes with exactly the
  same queries; `--config` works (it failed with a type error).

## [0.1.1] - 2026-11-27

Found by installing 0.1.0 on fresh Ubuntu 24.04 and 22.04 machines, step by step as
the README says:

- Scans and photos indexed before Tesseract was installed are read again once it is
  (they were never retried, because the files had not changed). The index records
  whether OCR was available for each file; upgrading marks the scans that had no text.
- Python 3.10 is supported, so the stock `pipx` on Ubuntu 22.04 installs filesonar
  (it failed with "No matching distribution found").
- The README says how to get `pipx` and Ollama on a fresh system.
- `filesonar uninstall` undoes setup: stops and removes the service (otherwise it would
  keep restarting after `pipx uninstall`), unregisters from OpenClaw and Claude Code,
  removes the skill and revokes Google Drive access. It keeps the index and settings
  unless asked (`--delete-data`), and then deletes only files filesonar created.
- On a server without a desktop, setup explains how to keep the service running after
  logout (`loginctl enable-linger`).
- Tested on fresh Fedora 45 (Python 3.15) and Arch (Python 3.14); CI covers 3.10 to 3.14.
- `filesonar setup` shows model download progress that only goes up (it restarted at
  0% for each part of the model).

## [0.1.0] - 2026-11-27

First release.

- `filesonar setup` takes a fresh install to a working one: checks Ollama and downloads
  the models, checks Tesseract, writes the config with the folders you choose, connects
  Google Drive if you want, registers with OpenClaw (and installs its skill) and Claude
  Code, installs a systemd user service and starts the first index. Safe to run again;
  it only changes what is missing.
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
- Google Drive support (opt-in): `filesonar drive login` signs in with your own OAuth
  client; Drive files, Google Docs, Sheets and Slides are indexed locally and kept up to
  date through the Drive change feed (`filesonar watch` polls every `poll_minutes`).
  Files that also exist on this computer are matched by MD5 and never downloaded;
  photos are described from Drive's preview image.
- Search results from Drive carry their Drive folder and a private link; `--source`
  / `source` limits results to `local` or `drive`.
- `share_file` MCP tool: a private Drive link for a laptop file, uploading it to a
  `filesonar` folder only if Drive does not already have it.
- Signing in to Drive again needs no client_secret file: the saved OAuth client is reused.
- `filesonar doctor` warns when Ollama has a model only partly in GPU memory.
- `scripts/eval_search.py` measures search quality on your own index; the fusion
  constant (5) was chosen with it on a real 16,000-file index.
