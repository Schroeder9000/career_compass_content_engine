# Career Compass AI - Content Engine

Multi-agent workflow for producing Career Compass blog content with OpenAI + LangChain, grounded by a lightweight RAG layer and deployable on Modal for weekly runs.

## Features
- **Supervisor-led workflow**: orchestrates SEO research, writing, image prompt generation, and Webflow publishing.
- **RAG grounding**: uses local examples to maintain tone, structure, and visual style.
- **Modal scheduling**: weekly automation via `modal_app.py`.
- **Webflow publishing hook**: prepared to POST content to a Webflow collection via API.

## Project layout
- `src/career_compass_content_engine/`
  - `agents/`: modular agents (research, writing, image design, publishing)
  - `rag.py`: builds a FAISS vector store from example content
  - `supervisor.py`: orchestrates the end-to-end workflow
  - `run_workflow.py`: CLI entrypoint for local runs
  - `modal_app.py`: Modal app definition with weekly schedule
  - `data/examples/`: seed examples to ground writing and visuals

## Setup
1. Python 3.11+ recommended.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Copy `.env.example` to `.env` and populate:
   ```bash
   OPENAI_API_KEY=...
   WEBFLOW_API_TOKEN=...            # optional until ready to publish
   WEBFLOW_SITE_ID=...              # optional
   WEBFLOW_COLLECTION_ID=...        # optional
   WEBFLOW_AUTHOR_ID=...            # optional reference field for author

   MODAL_SECRET_NAME=career-compass-secrets  # optional if using Modal secrets
   ```
4. Add prior blog samples to `src/career_compass_content_engine/data/examples/text/` and visual prompts/alt-text to `data/examples/images/`. Supported file types: `.md`, `.txt`, `.json`.
   - RAG ingest automatically reads both folders, so drop in representative copy and image prompt JSON/Markdown to keep tone and visuals consistent.

## Running locally
```bash
python -m career_compass_content_engine.run_workflow
```
The CLI prints the research recommendation, generated blog markdown, image plan, and (if configured) Webflow publishing result.

## Modal deployment
- Build a Modal image including this repo and secrets.
- Ensure `OPENAI_API_KEY` (and Webflow secrets if publishing) are available via `modal.Secret` or environment variables.
- Deploy the scheduled function defined in `modal_app.py`:
  ```bash
  modal deploy src/career_compass_content_engine/modal_app.py
  ```

## Notes on Webflow
Publishing is a best-effort HTTP call. If `WEBFLOW_API_TOKEN` or `WEBFLOW_COLLECTION_ID` are not provided, publishing is skipped with a status message so local iteration is safe.
