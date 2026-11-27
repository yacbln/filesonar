# Security policy

filesonar reads people's personal files, so security reports are taken seriously and
handled before anything else.

## Reporting a vulnerability

Please **do not open a public issue**. Use GitHub's private vulnerability reporting
("Security" tab → "Report a vulnerability") with:

- what an attacker can do, and under which conditions
- steps or a file that reproduces it
- the filesonar version (`filesonar --version`) and your OS

You will get an acknowledgement within a few days. Fixes are released as soon as they
are ready, and reporters are credited unless they prefer otherwise.

## What counts

In scope, for example:

- a file outside the configured folders, or on the built-in exclusion list, ending up in
  the index or in a tool result
- the MCP server reading, revealing or changing anything beyond what is documented
- file contents or queries being sent anywhere other than the configured model server
  (the speech model is downloaded once from Hugging Face during setup; nothing is sent)
  (or, with Google Drive enabled, Google's Drive API)
- the Google sign-in token leaking (logs, file permissions, other hosts), or `share_file`
  uploading anything other than the indexed file it was asked for, or outside its folder
- a crafted file that crashes the indexer in a way that stops other files from being
  indexed, or that makes filesonar execute code
- the index being readable by other local users

Out of scope: an agent misbehaving because of text inside a user's own file
(filesonar marks that content as untrusted, but cannot control the agent), and issues
that require the attacker to already control the user's account or config file.

## Design notes

The [README](README.md#privacy-and-security) lists the protections in place: loopback-only
model host by default, a non-overridable list of sensitive paths, no symlink following,
an MCP server that only answers from the index and only writes copies of indexed files
into its own outbox folder, opt-in Google Drive access that contacts only Google and
stores its token with private permissions, private file permissions, and
limits on image size, PDF pages and file size.
