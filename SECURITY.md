# Security Policy

## Supported versions

Sift is pre 1.0. Only the latest release on `main` receives fixes.

## Reporting a vulnerability

Report privately through
[GitHub Security Advisories](https://github.com/itstheraj/sift-search/security/advisories/new).
Please do not open a public issue for a vulnerability.

Expect an acknowledgement within seven days. If a fix is warranted, it lands on
`main` and is credited in the release notes unless you ask otherwise.

## Threat model

Sift indexes files you point it at and stores everything in a SQLite database
under `~/.local/share/sift`. It makes no network calls except to download model
weights from Hugging Face on first use. There is no telemetry and no cloud
component.

The things worth reporting:

- Indexed content leaking outside the local machine.
- A crafted file that achieves code execution during extraction, OCR, or
  transcription. Sift parses untrusted PDFs, Office documents, HTML, images, and
  media through third party libraries.
- The KRunner D-Bus service or the Dolphin service menu doing something a caller
  should not be able to trigger.
- Model weights being fetched from somewhere other than the configured source.

The things that are not vulnerabilities:

- Sift indexing a file you told it to index.
- The index containing text from your own documents. It is a search index. Treat
  `~/.local/share/sift` as being as sensitive as the folders you indexed.
