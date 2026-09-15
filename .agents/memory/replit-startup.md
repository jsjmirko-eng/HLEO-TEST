---
name: Replit startup configuration
description: Replit-specific constraints for running HLEO's FastAPI root on port 8000.
---

For HLEO's FastAPI root, keep the Uvicorn command on port 8000 and configure the Replit workflow as a console workflow waiting for port 8000; the port mapping belongs in `.replit`.

**Why:** Replit rejects direct edits to `.replit`; it requires schema validation and replacement through the dedicated configuration callback. Package installation can also rewrite `requirements.txt`, so the diff must be checked and generated duplicates or extras removed.

**How to apply:** Use the validated `.replit` replacement flow, then configure or restart the single `Start application` workflow and verify both `/` and `/health`.