# filesonar

[![CI](https://github.com/yacbln/filesonar/actions/workflows/ci.yml/badge.svg)](https://github.com/yacbln/filesonar/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/yacbln/filesonar/blob/main/LICENSE)

**Search your files by what is inside them — privately, on your own machine — and let your AI agent do it for you.**

Your file manager searches by file name. That is fine until you need "the PDF with the car insurance renewal" or "that photo of the red bike by the lake" and have no idea what either file is called. filesonar reads your documents, PDFs, scans, screenshots and photos once, remembers what they contain, and answers questions in plain language:

```console
$ filesonar search "automobile coverage"
1. /home/sam/Documents/insurance/auto-policy.pdf p.1
   pdf, 2024-03-02, matched in text
   Car insurance policy. Vehicle: 2019 Toyota Corolla. Premium 640 EUR per year.
```

Notice the query shares no words with the file. filesonar combines classic keyword search with meaning-based search, so both "invoice 4471" and "that bill from the hardware store" work.

It also runs as an [MCP](https://modelcontextprotocol.io) server, so an agent such as [OpenClaw](https://openclaw.ai), Claude Code or Cursor can search your files when you ask it to — from your phone, if your agent lives in Telegram or WhatsApp.

Everything runs locally through [Ollama](https://ollama.com). No file, no snippet and no query ever leaves your computer, unless you point filesonar at a model server elsewhere or turn on Google Drive.

Optionally, it can also index your [Google Drive](#google-drive) — the files are downloaded and understood on your machine, just like local ones — and hand out a private Drive link to a file on your laptop.

## What it understands

| Files | How |
|---|---|
| PDF | Text per page; pages that are scans are read with OCR |
| Word (`.docx`) | Paragraphs, tables, title and author |
| Images (`.jpg`, `.png`, `.webp`, `.gif`, `.bmp`, `.tiff`) | A local vision model describes what the picture shows; OCR reads any text (receipts, screenshots, slides); date and camera from EXIF |
| Text (`.txt`, `.md`, `.csv`, `.json`, `.html`, …) | As is |

With [Google Drive](#google-drive) enabled, the same goes for files in your Drive, plus Google Docs, Sheets and Slides. Coming next: audio and video. See the [roadmap](#roadmap).

## Install

You need Linux with Python 3.10 or newer (Ubuntu 22.04 and later, Debian 12, Fedora, Arch, …) and [Ollama](https://ollama.com/download), which runs the models on your computer:

```bash
curl -fsSL https://ollama.com/install.sh | sh     # if you do not have Ollama yet
sudo apt install pipx && pipx ensurepath           # if you do not have pipx yet (Fedora: dnf, Arch: pacman)
```

Open a new terminal so `pipx ensurepath` takes effect, then:

```bash
pipx install filesonar
filesonar setup
```

`uv tool install filesonar` works too, if you use [uv](https://docs.astral.sh/uv/).

`setup` asks a few questions and does the rest. Each step checks first and only changes what is missing, so it is safe to run again — and running it after `pipx upgrade filesonar` is how an upgrade takes effect. It:

1. checks that Ollama runs, and downloads the text model `nomic-embed-text` (274 MB). It offers the vision model `qwen2.5vl:7b` that describes photos (about 6 GB, best with a GPU);
2. checks for [Tesseract](https://github.com/tesseract-ocr/tesseract), which reads text in scans and screenshots, and prints the install command if it is missing (`sudo apt install tesseract-ocr`; add `tesseract-ocr-fra`, `-ara`, … for other languages). Setup never runs `sudo` itself;
3. writes `~/.config/filesonar/config.toml` with the folders you choose (by default Documents, Pictures, Downloads and Desktop, if they exist);
4. optionally connects [Google Drive](#google-drive);
5. registers filesonar with OpenClaw and installs its skill, and with Claude Code if you have it;
6. installs a systemd user service that keeps the index up to date;
7. starts the first index.

For a setup without questions: `filesonar setup --yes`, optionally with `--folder ~/Papers` (repeatable), `--no-vision`, `--no-service`. See `filesonar setup --help`.

A GPU makes photo descriptions fast. Text, PDFs and OCR are fine on a CPU.

To install the latest code from GitHub instead: `pipx install git+https://github.com/yacbln/filesonar`.

## Quick start

```bash
filesonar search "lease agreement for the apartment"
filesonar status        # how much is indexed
filesonar doctor        # checks Ollama, the models, Tesseract and Drive
```

Useful search options: `--kind pdf|document|image|text`, `--folder ~/Documents/taxes`, `--after 2024-01-01`, `--before 2024-12-31`, `--source local|drive`, `--json`.

### Keep the index up to date

`filesonar watch` re-indexes files as soon as they are added, edited, moved or deleted (it waits for a few quiet seconds, so copying 500 photos results in one pass, not 500). It also does a full scan once a day as a safety net. Moving or renaming a file is recognised by its content and costs nothing.

`filesonar setup` runs it in the background as the systemd user service `filesonar-watch`:

```bash
systemctl --user status filesonar-watch
journalctl --user -u filesonar-watch -f     # see what it is doing
```

## Use it from an AI agent

`filesonar serve` speaks MCP over stdio. Three tools are read-only; `prepare_file` writes, but only copies into its own outbox folder; `share_file` exists only when Google Drive is enabled:

| Tool | What it does |
|---|---|
| `search_files` | Ranked results with path, kind, date, the passage that matched and why it matched (text, OCR, caption or metadata). Identical copies of a file are one result, with the other locations listed. Filters for kind, folder, date range and source (`local` or `drive`). Drive results include their Drive folder and a private link. |
| `file_info` | Details and a short text preview for a file **that is already in the index**. It cannot be used to open arbitrary paths. |
| `prepare_file` | Copies a file **that is already in the index** into the outbox folder (by default OpenClaw's media folder, `~/.openclaw/media/filesonar`) and returns the copy's path, so a chat agent can attach it to a message. Copies are private and deleted after 24 hours. Works for Drive files too (downloaded, or taken from the identical local copy); Google Docs and Slides arrive as PDF, Sheets as `.xlsx`. |
| `share_file` | Returns a Google Drive link for an indexed laptop file. If the same file (same MD5) is already in your Drive, that link is returned and nothing is uploaded; otherwise a copy is uploaded to a `filesonar` folder in your Drive. Links are private: only you can open them. Handy for files too large to attach. |
| `index_status` | How many files are indexed and when the index last changed. |

Why `prepare_file`? Chat agents usually may only view or send files inside their own folders. Without it, an agent asked to "send me that photo" will try to run `cp`, a shell command that could write anywhere and so needs your approval each time. `prepare_file` does the one safe copy itself, and because it is marked as non-destructive and local, OpenClaw runs it without a prompt.

### OpenClaw

```bash
openclaw mcp add filesonar --command filesonar --arg serve
openclaw mcp probe filesonar
```

`filesonar setup` does this for you, and also installs a [skill](https://github.com/yacbln/filesonar/blob/main/src/filesonar/skill/SKILL.md) that teaches OpenClaw when to use these tools (into `~/.openclaw/workspace/skills/filesonar/`). If your gateway runs elsewhere or hot reload is off, reload or restart it afterwards. If `openclaw` on your `PATH` is not the one your gateway runs, point setup at the right one: `filesonar setup --openclaw /path/to/openclaw`.

### Claude Code

```bash
claude mcp add filesonar -- filesonar serve
```

### Any other MCP client

```json
{
  "mcpServers": {
    "filesonar": { "command": "filesonar", "args": ["serve"] }
  }
}
```

Add `"--config", "/path/to/config.toml"` before `"serve"` if your config is not in the default place.

## Google Drive

filesonar can index the files in your Google Drive and keep up with changes, so "find the lease" also finds the copy you only have in Drive. File contents are downloaded to your computer and processed there by the same local models; only Drive's own API is contacted. Photos are described from Drive's preview image (no need to download a 6 MB original), and files that are already on your computer — say, photos synced to Drive — are recognised by their checksum and never downloaded at all.

After the first sync, filesonar uses Drive's [change feed](https://developers.google.com/drive/api/guides/manage-changes): `filesonar watch` asks "what changed?" every few minutes, which is one cheap request when nothing did. The Drive API is free for this kind of personal use.

### One-time setup (about 5 minutes)

filesonar does not ship a shared Google app; you create your own, so your access is yours alone and does not depend on anyone else's project.

1. Open [Google Cloud Console](https://console.cloud.google.com/) and create a project (any name, e.g. "filesonar").
2. **APIs & Services → Library**: search for **Google Drive API** and enable it.
3. **Google Auth Platform** (formerly *OAuth consent screen*): click **Get started**, enter an app name (e.g. "filesonar") and your email, and choose **External**.
4. **Audience**: add your own Google account under **Test users**. Apps in *Testing* get sign-ins that expire after 7 days (filesonar then asks you to run `filesonar drive login` again). To avoid that, fill in **Branding** — app name, support email, a homepage and a privacy policy URL; this repository and its [`PRIVACY.md`](https://github.com/yacbln/filesonar/blob/main/PRIVACY.md) are fine — and click **Publish app** so the app is *In production*. You do not need Google's verification for your own use.
5. **Clients → Create client**, application type **Desktop app** (not *Web application*, and leave *"Use this client for an AI-powered agent"* unticked). Download the JSON file and keep it private.
6. Sign in and turn Drive on:

   ```bash
   filesonar setup --drive-client-secret ~/Downloads/client_secret_XXXX.json
   ```

   Your browser opens Google's consent page. Because the app is yours and unverified, Google shows *"Google hasn't verified this app"*: choose **Advanced → Go to filesonar**. Allow both permissions (read your Drive, and create files filesonar makes). Setup then turns Drive on in your config, and the background service indexes it. (By hand: `filesonar drive login --client-secret FILE`, set `enabled = true` under `[drive]`, run `filesonar index`.) You can delete the downloaded JSON afterwards; filesonar keeps what it needs in its own private file, and to sign in again later (for example after publishing the app) plain `filesonar drive login` is enough.

The sign-in is saved to `~/.local/share/filesonar/drive-token.json` (permissions `0600`). `filesonar drive logout` revokes it and removes Drive files from the index.

### Commands and options

```bash
filesonar drive status     # who is signed in, how many Drive files are indexed
filesonar drive sync       # fetch changes now (`--full` re-lists everything)
filesonar search "visa appointment" --source drive
```

```toml
[drive]
enabled = true
folders = ["Taxes", "Photos/2024"]   # only these folders in My Drive; empty = all of it
exclude = ["/Archive/**"]            # patterns, matched against the path inside My Drive
include_shared = false               # also index files others shared with you
share_folder = "filesonar"           # where share_file uploads to
poll_minutes = 5
image_previews = true                # describe photos from Drive's preview image
```

The built-in exclusions (keys, `.env` files, password stores, …) apply to Drive too.

## Configuration

All options with their defaults (the `[drive]` section is described under [Google Drive](#google-drive)):

```toml
roots = ["~/Documents", "~/Pictures", "~/Downloads"]   # folders to index

# Extra patterns to skip, matched against the path inside each folder.
# "*" stays within one folder, "**/" spans any number of folders,
# a leading "/" anchors to the indexed folder itself.
exclude = []                  # e.g. ["**/old-backups/**", "*.log"]

# data_dir = "~/.local/share/filesonar"   # where the index is stored
# outbox_dir = "~/.openclaw/media/filesonar"   # where prepare_file puts copies

[models]
host = "http://127.0.0.1:11434"
embedding_model = "nomic-embed-text"
vision_model = "qwen2.5vl:7b"
caption_images = true
allow_remote_host = false     # see "Privacy and security"
timeout_seconds = 120

[index]
max_file_size_mb = 50
max_pdf_pages = 300
max_chunks_per_file = 300     # very long books and data exports: only the start is indexed
ocr = true
ocr_languages = "eng"         # Tesseract codes joined by "+", e.g. "eng+fra+ara"
include_hidden = false        # dot-files and dot-folders
default_excludes = true       # .git, node_modules, venvs, build output, caches, trash, ...
skip_dependency_folders = true  # folders where software packages were installed
chunk_chars = 1200
chunk_overlap = 200

[search]
max_distance = 0.44           # how close a meaning-only match must be (lower = stricter)
relative_margin = 0.12
```

Your `exclude` patterns are added to the defaults rather than replacing them. Folders where Python packages were installed (they contain `*.dist-info` entries, e.g. a bundled AWS Lambda) are skipped too, so a code project inside `~/Documents` does not bury your own files under thousands of library files.

Unknown options are rejected instead of silently ignored, so a typo never turns a setting off.

If you switch `embedding_model`, rebuild the index with `filesonar index --rebuild`; filesonar refuses to mix vectors from different models. `max_distance` is tuned for `nomic-embed-text` — other embedding models may need a different value.

## Privacy and security

filesonar reads your personal files, so it is built to be boring and predictable:

- **Local only by default.** It talks to one server: the Ollama address in your config, which must be on this machine unless you set `allow_remote_host = true`. It ignores `HTTP_PROXY`-style variables for that, has no telemetry and makes no other network requests.
- **Google Drive is opt-in and contacts only Google.** With `[drive] enabled = true`, filesonar talks to `googleapis.com` (the Drive API and sign-in) and to the `googleusercontent.com` preview links the Drive API hands out — nothing else. Drive files are processed on your machine, never sent anywhere. Sign-in uses Google's flow for installed apps (PKCE, random `state`, a one-shot listener on `127.0.0.1`); the refresh token lives in a `0600` file and is never logged. Uploads happen only through `share_file`, only for indexed files, only into one folder that filesonar created, and the Drive permission filesonar asks for (`drive.file`) cannot change or delete files it did not create.
- **Secrets are never indexed.** SSH and GPG keys, password stores, cloud credentials, browser profiles, crypto wallets, `.env` files, `*.pem`/`*.key` files and similar are always skipped. This list is built in and cannot be switched off by the config.
- **No symlink surprises.** Symbolic links are never followed, and a file whose real location is outside your indexed folders is skipped.
- **The agent only sees the index.** The MCP server never opens a path that is not in the index, returns short snippets (and at most a 2,000-character preview), and tells the agent to treat file contents as data rather than instructions — a defence against documents that try to give orders to AI assistants.
- **`prepare_file` can only copy, and only into its outbox.** It copies files that are in the index and still allowed by your config, reads them without following symlinks, writes `0600` copies into `0700` folders inside `outbox_dir`, never writes through a symlink, and only ever deletes its own copies older than 24 hours. Your original files are never modified. The outbox is never indexed.
- **GPS coordinates are not stored.** Photo dates and camera models are, locations are not.
- **The index is private.** It is a single SQLite file created with `0600` permissions inside a `0700` folder. Delete it at any time; it can be rebuilt from your files.
- **Defensive parsing.** Images that would decode to more than ~90 megapixels, password-protected PDFs, oversized files and binary files with text extensions are skipped rather than processed.

Found a problem? Please see [SECURITY.md](https://github.com/yacbln/filesonar/blob/main/SECURITY.md). The full privacy policy is in [PRIVACY.md](https://github.com/yacbln/filesonar/blob/main/PRIVACY.md).

## How it works

```
 files ──► extract ──────────────► chunk ──► embed (Ollama) ──► SQLite
           PDF/DOCX text                                         ├─ FTS5 keyword index
           OCR (Tesseract)                                       └─ sqlite-vec vectors
           photo captions (vision model)
           EXIF / document metadata
                                                    query ──► keyword hits ─┐
                                                          └─► nearest by meaning ─┴─► fused ranking ──► results
```

- Changes are detected by size and modification time first and confirmed with a SHA-256 of the content, so unchanged files are never read twice and a moved file keeps its index entry.
- Keyword and meaning results are merged with reciprocal rank fusion. Matches by meaning only count if they are genuinely close; otherwise unrelated searches would always "find" something.
- If Ollama is not running, search keeps working with keywords only and says so.

## Performance

Embedding text is near-instant (well under 0.1 s per chunk on a desktop GPU). Describing a photo is the slow step, and it is much faster when the vision model fits entirely in video memory. If your GPU is busy with other models — for example the model your agent runs on — Ollama splits the vision model between GPU and CPU and captions slow down to several seconds per photo. Options:

- run the first big index when the GPU is free (overnight), or
- use a smaller vision model such as `qwen2.5vl:3b`, or
- set `caption_images = false`; photos stay searchable by any text in them, their date and name.

Images that could not be captioned are retried automatically the next time the vision model is available.

## Troubleshooting

- **`filesonar doctor`** shows what is missing.
- **"meaning-based search unavailable"** — Ollama is not running or the embedding model is not pulled.
- **Photos suddenly take 10+ seconds each** — Ollama loaded the vision model while video memory was short and split it between GPU and CPU; it stays split until reloaded. `filesonar doctor` warns about this. Free some GPU memory and run `ollama stop qwen2.5vl:7b`; the next photo loads it fully onto the GPU.
- **A file is not found** — check that its folder is in `roots`, that it is not excluded or hidden, and `filesonar status` for errors. `filesonar -v index` prints every file it reads.

## Roadmap

- [x] Documents, PDFs (with OCR), images (captions + OCR), text
- [x] Hybrid keyword + meaning search, MCP server, folder watcher
- [x] Google Drive: index files in Drive and keep them in sync (Drive Changes API)
- [x] `share_file`: a private Drive link to a laptop file, uploading only if it is not in Drive yet
- [ ] Audio and video (transcripts and key frames)
- [ ] More formats: `.odt`, `.pptx`, `.xlsx`, `.epub`
- [ ] macOS support

Ideas and pull requests are welcome.

## Contributing

See [CONTRIBUTING.md](https://github.com/yacbln/filesonar/blob/main/CONTRIBUTING.md). The short version:

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
pytest              # fast suite, no models needed
pytest -m ollama    # end-to-end against your local Ollama
ruff check . && ruff format --check .
```

## License

[MIT](https://github.com/yacbln/filesonar/blob/main/LICENSE)
