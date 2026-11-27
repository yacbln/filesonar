# Privacy policy

filesonar is software you run on your own computer. There is no filesonar service,
account or server, and the people who write filesonar never receive any of your data.

## What filesonar reads

- Files in the folders you list in its configuration.
- If you turn on Google Drive and sign in: the files in your Google Drive (or the Drive
  folders you choose), their names, folders, dates, sizes and checksums.

## What it does with it

Everything is processed on your computer, by the AI models you run there (Ollama). The
result — extracted text, descriptions of photos, and metadata — is stored in a private
index file in your own data folder (`~/.local/share/filesonar/` by default), readable
only by your user account. You can delete it at any time.

filesonar does not:

- send your files, the index or your searches to its authors or to any third party;
- include analytics, telemetry or crash reporting;
- store GPS locations from photos;
- index secrets such as SSH keys, password stores, browser profiles or `.env` files.

## Google Drive

When you enable Google Drive, filesonar uses an OAuth client that **you** create in your
own Google Cloud project, and asks Google for two permissions:

- `drive.readonly` — to list and read your Drive files so they can be indexed locally;
- `drive.file` — to create the one folder and the files that `share_file` uploads when
  you ask for a link. This permission cannot change or delete files filesonar did not
  create.

Your sign-in (a refresh token) is stored only on your computer, in a file readable only
by you. filesonar talks only to Google's own servers for this. Running
`filesonar drive logout` revokes the access and removes Drive data from the index; you
can also revoke it at <https://myaccount.google.com/permissions>.

filesonar's use and transfer of information received from Google APIs adheres to the
[Google API Services User Data Policy](https://developers.google.com/terms/api-services-user-data-policy),
including the Limited Use requirements.

## AI agents

If you connect filesonar to an AI agent (OpenClaw, Claude Code, Cursor, ...), that
agent receives search results and short excerpts from your files when it asks for them.
What the agent then does with them — including sending them to the AI model provider
it uses — is governed by that agent's and that provider's privacy policies, not this
one.

## Contact

Questions or concerns: open an issue in the project's repository.
