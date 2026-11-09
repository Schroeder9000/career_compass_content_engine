# Career Compass AI - Content Engine

Multi-agent workflow for producing Career Compass blog content with OpenAI + LangChain, grounded by a lightweight RAG layer and deployable on Modal for weekday runs.

## Features
- **Supervisor-led workflow**: orchestrates SEO research, writing, image prompt generation, and Webflow publishing.
- **RAG grounding**: uses local examples to maintain tone, structure, and visual style.
- **Modal scheduling**: weekday automation via `modal_app.py`.
- **Webflow publishing hook**: prepared to POST content to a Webflow collection via API.

## Project layout
- `src/career_compass_content_engine/`
  - `agents/`: modular agents (research, writing, image design, publishing)
  - `rag.py`: builds a FAISS vector store from example content
  - `supervisor.py`: orchestrates the end-to-end workflow
  - `run_new_blog_workflow.py`: CLI entrypoint for local runs
  - `modal_app.py`: Modal app definition with weekday schedule
  - `data/examples/`: seed examples to ground writing and visuals

## Setup
1. Python 3.12+ recommended.
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

   PUBLISH_MODE=live                # or draft: saves without publishing
   ```
4. Add prior blog samples to `src/career_compass_content_engine/data/examples/text/` and visual prompts/alt-text to `data/examples/images/`. Supported file types: `.md`, `.txt`, `.json`.
   - RAG ingest automatically reads both folders, so drop in representative copy and image prompt JSON/Markdown to keep tone and visuals consistent.

## Running locally
```bash
python -m career_compass_content_engine.run_new_blog_workflow
```
The CLI prints the research recommendation, generated blog markdown, image plan, and (if configured) Webflow publishing result.

## Modal deployment
- Build a Modal image including this repo and secrets.
- Ensure `OPENAI_API_KEY` (and Webflow secrets if publishing) are available via `modal.Secret` or environment variables.
- Deploy the scheduled function defined in `modal_app.py`:
  ```bash
  modal deploy src/career_compass_content_engine/modal_app.py
  ```

## Publication safety

The deployed app uses the existing `careeer_compass_langchain_content_engine`
Modal secret and runs Monday–Friday at 7 a.m. America/New_York. A passing article
publishes only its CMS item; the engine has no site-wide publish operation.
Website releases are a separate Webflow action.

`PUBLISH_MODE=draft` saves an approved article without publishing it. Invalid
modes, mismatched CMS schemas, uncertain create results, duplicate slugs, and
required missing images stop writes or publication rather than guessing.
Non-idempotent creates are never automatically retried after ambiguous failures.

The optional `publication-date` DateTime field stores the public article date
independently of Webflow's update and republish timestamps. Live articles receive
a UTC date when created. For manual drafts, set Publication date when publishing;
preserve it during later maintenance. If the field exists with an incompatible
type, publication stops for review.

Editorial scores are validated and averaged in code. Below-threshold results,
critical issues, and missing/negative metadata approval cannot publish. After
three unsuccessful editorial evaluations, or failed metadata QA, the draft and
reason are retained in `.cache/blocked/` on the existing Modal cache volume.
Existing alerts distinguish success, draft, blocked, skipped, and error outcomes.

Current product claims live in `data/product_facts.json` inside the package.
The owner's September 24, 2026 copy preference is included there: omit minimum-age
and age-eligibility qualifiers from public article copy and metadata. This applies
to research, writing, editing, evaluation, and publishing prompts; it does not
change app eligibility or legal policy.
Writing, editing, evaluation and metadata prompts share those facts as an
accuracy reference. Research uses the broader editorial mission below.
Historical examples are voice references, not current product documentation.
Update the website copy and this facts file together when the offer changes.

## Editorial coverage

The blog serves early-career professionals across first jobs and transitions,
communication and workplace relationships, skills and learning, work habits and
boundaries, confidence and belonging, career direction and job search, workplace
change and AI literacy, and advancement and pay. `editorial_policy.py` chooses an
under-covered pillar from successful publication history and records it on each
new run. Drafts, failed runs, and duplicate publication records do not consume a
coverage slot. Legacy promotion/pay titles are recognized during the transition.
Advancement/pay is capped at two of the next rolling ten publications and cannot
run consecutively. Search queries follow the assigned pillar.

Brand mentions, app links and sales CTAs are optional. Broad career articles stand
on their own; a directly relevant advancement/pay article may include at most one
short product paragraph of one or two sentences, no more than 60 words. Product
facts are never a checklist to insert. The editorial brief follows the article
through writing, editing, evaluation and metadata. Deterministic checks reject
promotion/pay topic pivots and product pitches in broad articles. Failed closing
repairs block publication and preserve the draft instead of appending a sales
paragraph. The weekday schedule, current model, image optimization and publication
safeguards remain unchanged.

## Google reporting credentials

Use `GOOGLE_SERVICE_ACCOUNT_JSON` for inline service-account JSON in a Modal
secret. `GSC_SERVICE_ACCOUNT_JSON` still accepts a local/mounted JSON file path
or inline JSON. Never commit a key file or bake it into an image. Both clients
use the existing reporting identity with read-only scopes.

Production properties: `GSC_SITE_URL=sc-domain:careeraiapp.com` and
`GA4_PROPERTY_ID=427074708` (website, not the mobile app property).
Missing reporting inputs are reported as unavailable, not successful data loads.

## Verification

### Image delivery

Generated artwork is converted to WebP at quality 90 (minimum 86), capped at
1600 pixels wide and 1 MB. Pillow with WebP support is required. Failed decoding,
compression, schema validation, or uploads stop publication; original oversized
PNGs are never used as a fallback. Alpha, aspect ratio, orientation, and color
profiles are preserved, and smaller sources are never enlarged.

The publisher creates 500, 800, and 1080 pixel variants where the source permits,
plus the full image. The thumbnail field uses the 800 pixel version. Webflow must
have optional PlainText fields `featured-image-srcset` and
`thumbnail-image-srcset`; the publisher fills these with hosted URLs and width
descriptors. Optional Number fields `featured-image-width`,
`featured-image-height`, `thumbnail-image-width`, and `thumbnail-image-height`
store the corresponding dimensions so templates can reserve space before load.
The article, blog, category, and author templates use CMS-bound
image embeds with explicit `srcset` and `sizes`, because API-uploaded images do
not automatically receive responsive variants in the published CMS markup.
Article hero images load eagerly at high priority; article cards load lazily.

For a manually replaced CMS image, update its matching srcset field as well, or
clear that field so the image falls back safely to the new CMS source URL.

### Automated checks

Run `python -m pytest`. Tests mock network/model calls and exercise the compiled
workflow, rejection paths, preserved CMS fields, and item-only/draft publication.

`PYTHONPATH=src modal run scripts/verify_editorial.py --output /tmp/editorial-preview.json`
runs the tests in Modal, reads recent history, requests four topic proposals and
previews one broad article through writing, editing, evaluation and metadata.
It uses the existing OpenAI secret without exposing it, does not publish or write
to the CMS, and verifies that production topic history is unchanged. Model
evaluation and metadata results are included in the local receipt for review.

`PYTHONPATH=src modal run scripts/verify_launch.py` runs tests in the Modal image
and creates one clearly named, unpublished launch verification draft. It reuses
an existing editorial image, does not invoke OpenAI, and does not publish a site
or an article. Repeated verification reuses the same draft after checking it is
still unpublished. Leave it as a draft for audit evidence.
