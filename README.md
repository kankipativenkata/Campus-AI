# Campus AI Platform

One open-source, role-based AI assistant for a university, built with **Streamlit** and **Gemma 4**.
Students, faculty and admins log in to the same app and see only the modules their role allows.
Any university can adopt it by editing config files and uploading its own documents.

> **All bundled sample data is fictional** ("Demo University"). Replace it with real documents before real use.

| Role | Modules |
|---|---|
| **Everyone** | Campus Knowledge Assistant (RAG chat with cited sources) |
| **Student + Faculty** | Snap-and-Explain (photo to explanation, English/Telugu), Diagram Reviewer |
| **Faculty** | Answer-Script Feedback Assistant, CO-PO Mapper and Question Paper Auditor |
| **Admin** | Knowledge Base Manager (upload data for RAG), Config Manager, User Management, Usage/Audit/Health |

---

## Table of contents
1. [How it works](#1-how-it-works)
2. [Project structure](#2-project-structure)
3. [Prerequisites](#3-prerequisites)
4. [Get a Gemini API key](#4-get-a-gemini-api-key-for-gemma-4)
5. [Run locally (step by step)](#5-run-locally-step-by-step)
6. [Log in and try every module](#6-log-in-and-try-every-module)
7. [Test data files (what each one is for)](#7-test-data-files)
8. [Run the automated tests](#8-run-the-automated-tests)
9. [Evaluate RAG quality](#9-evaluate-rag-quality)
10. [Deploy to Streamlit Community Cloud (live URL)](#10-deploy-to-streamlit-community-cloud-live-url)
11. [Database options](#11-database-options)
12. [Local models with Ollama (no API)](#12-local-models-with-ollama-no-api)
13. [Docker](#13-docker)
14. [Configuration reference](#14-configuration-reference)
15. [Adapting it for another university](#15-adapting-it-for-another-university)
16. [Adding a new module](#16-adding-a-new-module)
17. [Security and privacy notes](#17-security-and-privacy-notes)
18. [Troubleshooting](#18-troubleshooting)
19. [Known limitations (honest list)](#19-known-limitations)
20. [3-minute demo script](#20-3-minute-demo-script)
21. [Hackathon submission checklist](#21-hackathon-submission-checklist)

---

## 1. How it works

```
Streamlit UI (menu built from the user's role)
        |
  modules/*.py  <- each file = MANIFEST (name, roles) + render(user)
        |
  core/: auth | rag | llm | parsing | rubric | copo | diagram_checks | db | audit
        |
  Gemma 4 via Gemini API  (or local Ollama)        SQLite (default) or Postgres
  Embeddings via Gemini embedding model            chunks + vectors stored in the DB
```

**Design choices that matter**
- **Hybrid RAG:** BM25 keyword score + vector cosine similarity. With no API key it still works in keyword-only mode.
- **Grounded answers:** the model may answer only from retrieved passages; otherwise it says "not found". A similarity/keyword gate refuses out-of-scope questions *before* calling the model.
- **Role filter before retrieval:** faculty-only documents are excluded in the SQL query, so students can never retrieve them.
- **Human in the loop:** marks suggested by the AI are clamped to the rubric limits and need faculty approval before they are saved.
- **Model does extraction, code does judging:** diagram scoring uses fixed rules on the structure the model extracted, so scores are consistent.
- **Graceful degradation:** no key means the RAG chat, CO-PO mapper (rule-based) and diagram rule-checks still work.

---

## 2. Project structure

```
campus-ai/
├── app.py                      # entry point: login + role-based menu
├── requirements.txt            # runtime deps (used by Streamlit Cloud)
├── requirements-dev.txt        # + pytest, reportlab
├── Dockerfile, docker-compose.yml
├── .env.example                # local settings template
├── .streamlit/
│   ├── config.toml             # theme and upload limit
│   └── secrets.toml.example    # secrets template (Cloud and local)
├── config/                     # EDIT THESE to adapt to another university
│   ├── university.yaml         # name, tagline, RAG thresholds, suggested questions
│   ├── rubrics/*.yaml          # marking rubrics
│   ├── outcomes/*.yaml         # COs, POs, CO-PO matrix, Bloom's targets
│   └── prompts/*.yaml          # prompt packs per module
├── core/                       # shared services
│   ├── config.py db.py auth.py audit.py llm.py parsing.py
│   ├── rag.py rubric.py copo.py diagram_checks.py
│   ├── registry.py seed.py
├── modules/                    # one file per feature (plugins)
├── data/
│   ├── sample_docs/            # FICTIONAL docs auto-loaded on first run
│   └── test_data/              # files for you to upload while testing
├── scripts/                    # generate_test_data.py, eval_rag.py, reset_db.py
├── tests/                      # pytest suite (44 tests)
└── .github/workflows/ci.yml    # runs tests on every push
```

---

## 3. Prerequisites

- **Python 3.10 to 3.12** (3.11 recommended). Check: `python --version`
- **pip** (comes with Python)
- **Git** (only for deployment): `git --version`
- A **free Gemini API key** (section 4). The app starts without one, but Snap-and-Explain, Diagram Reviewer image mode, Answer-Script Feedback and AI-written RAG answers need it.
- Internet access for pip and the API.

---

## 4. Get a Gemini API key (for Gemma 4)

1. Open <https://aistudio.google.com> and sign in with a Google account.
2. Click **Get API key**, then **Create API key**.
3. Copy the key (starts with `AI...`). Treat it like a password: never commit it, never post it.
4. The free tier has rate limits. If you see `429` or `RESOURCE_EXHAUSTED` errors during a demo, wait a minute or enable billing on the key's project.

The app uses these model IDs by default (override in settings):

| Purpose | Default | Alternative |
|---|---|---|
| Text + vision | `gemma-4-26b-a4b-it` | `gemma-4-31b-it` |
| Embeddings | `gemini-embedding-001` | any embedding model your key can access |

Model names change over time. If you get "model not found", check the current list at <https://ai.google.dev/gemma/docs/core/gemma_on_gemini_api> and set `GEMMA_MODEL`.

---

## 5. Run locally (step by step)

### 5.1 Get the code
```bash
cd campus-ai          # the unzipped folder
```

### 5.2 Create a virtual environment
**macOS / Linux**
```bash
python3 -m venv .venv
source .venv/bin/activate
```
**Windows (PowerShell)**
```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
```
If PowerShell blocks the script: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`, then activate again.
You should see `(.venv)` in your prompt.

### 5.3 Install dependencies
```bash
pip install -r requirements-dev.txt
```
(Use `requirements.txt` if you do not need to run tests.)

### 5.4 Add your API key
Choose **one**:

**Option A: `.env` file (simplest)**
```bash
cp .env.example .env          # Windows: copy .env.example .env
```
Open `.env` and set `GEMINI_API_KEY=your-key`.

**Option B: Streamlit secrets file**
```bash
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
```
Edit it and set `GEMINI_API_KEY = "your-key"`.

Skipping this step is fine for a first look: the app runs in offline mode.

### 5.5 Start the app
```bash
streamlit run app.py
```
Your browser opens <http://localhost:8501>. On the **first run** the app creates `data/campus.db`, three demo users and loads the 8 sample documents (a few seconds; with a key it also creates embeddings).

### 5.6 Reset to a clean state
Stop the app (Ctrl+C), then:
```bash
python scripts/reset_db.py
```
The next run re-seeds everything.

---

## 6. Log in and try every module

Demo accounts (shown on the login page; hide them with `show_demo_credentials: false` in `config/university.yaml`):

| Role | Username | Password |
|---|---|---|
| Student | `student1` | `student123` |
| Faculty | `faculty1` | `faculty123` |
| Admin | `admin` | `admin123` |

Use a private window or **Sign out** to switch roles.

### 6.1 Admin: upload data for RAG
1. Log in as `admin`, open **Knowledge Base Manager**.
2. The 8 sample documents are already listed. Each shows `chunks` and `embedded` (embedded = number of chunks with vectors; 0 means keyword-only).
3. Upload a new file: choose a PDF/DOCX/TXT/MD/CSV/XLSX, set Category, **Audience** and Version, then **Upload and index**.
4. Upload another file with the **same title** and a higher version: the old version is archived automatically.
5. Use **Test a search** at the bottom to see scores for a query as a student/faculty/admin.

### 6.2 Student: ask questions
1. Log in as `student1`, open **Campus Knowledge Assistant**.
2. Click a suggested question or type your own, for example: *What is the minimum attendance for exam eligibility?*
3. Expand **Sources** to see document, page, version and the matching text.
4. Ask something not in the documents (*What is the capital of France?*): you get the "could not find" message.
5. Ask *How many casual leaves do faculty get?*: a student gets "not found" (faculty-only document), but `faculty1` gets the answer.
6. Open **Snap-and-Explain**, upload `data/test_data/snap/textbook_stack_page.png`, choose language, click **Explain this**.
7. Open **Diagram Reviewer**: see section 7 for the JSON (offline) and image (needs key) tests.

### 6.3 Faculty
1. Log in as `faculty1`.
2. **Answer-Script Feedback**: pick rubric `CSE1001_data_structures.yaml`, question Q3, upload `Q3_good.png` (or paste text from `Q3_good.txt`), click **Suggest marks**. Edit the "final" column if needed, then **Approve and save**. Download the CSV at the bottom.
3. **CO-PO Mapper**: choose **Use bundled sample**, click **Analyze paper**. Review tags, coverage charts, PO weights and the repeated-question table (Q1 and Q5 repeat from 2025). Download the Excel report.
4. **Config Manager** (read-only for faculty).

### 6.4 Admin extras
- **Config Manager**: edit YAML with validation (rubric marks must add up; matrix must refer to existing COs).
- **User Management**: add users, bulk import `data/test_data/users/users_bulk_import.csv`, deactivate users, reset passwords.
- **Usage, Audit & Health**: counts, feedback ratings, **unanswered questions** (content gaps), audit log, and **Test LLM call / Test embeddings** buttons to confirm your key works.

---

## 7. Test data files

All in `data/test_data/` (fictional/synthetic). Regenerate the images with `python scripts/generate_test_data.py`.

| File | Use it in | Expected result |
|---|---|---|
| `rag_test_questions.csv` | `scripts/eval_rag.py`, manual Q&A | 20 of 21 pass offline; the Telugu row needs embeddings |
| `answer_scripts/Q3_good.png` / `.txt` | Answer-Script Feedback, Q3 | high marks (about 9-10 of 10) |
| `answer_scripts/Q3_partial.png` / `.txt` | same | partial (about 4-6): defines stack, push only, one application |
| `answer_scripts/Q3_poor.png` / `.txt` | same | low (about 0-2): says FIFO and O(n) |
| `answer_scripts/Q5_good.png` / `.txt` | Feedback, question Q5 | high marks |
| `question_papers/question_paper_sample.txt` | CO-PO Mapper ("Use bundled sample" or upload) | 9 questions, 54 marks; Q1 and Q5 flagged as repeats |
| `question_papers/previous_paper_2025.txt` | auto-loaded as history | source of the repeats |
| `diagrams/flowchart_correct.png` | Diagram Reviewer, type Flowchart | score 100, no findings |
| `diagrams/flowchart_broken.png` | same | errors: no End, decision with 1 branch, orphan box |
| `diagrams/er_diagram.png` | type ER diagram | score 100 |
| `diagrams/*.json` | Diagram Reviewer, "From structure JSON" tab (works with no key) | same results as the images |
| `snap/textbook_stack_page.png` | Snap-and-Explain | explains stack, push/pop, LIFO |
| `snap/circuit_rc.png` | Snap-and-Explain | RC time constant = R x C = 10 ms |
| `snap/graph_parabola.png` | Snap-and-Explain | minimum of y = x^2 at the origin |
| `users/users_bulk_import.csv` | Admin, User Management, bulk import | 3 users created |

Sample documents in `data/sample_docs/` (loaded automatically; fictional): academic regulations (TXT), hostel rules (MD), fee structure (CSV), academic calendar (TXT), placement FAQ (MD, students), club directory (DOCX), scholarship policy (PDF), faculty leave policy (TXT, faculty only). They exist partly to test every parser.

**Judging AI output in tests:** AI marks and explanations can vary slightly between runs. Check the *pattern* (good > partial > poor), not exact numbers.

---

## 8. Run the automated tests

```bash
python -m pytest -q
```
Expected: **44 passed**. Tests use a temporary database, never call the network, and need no API key.

What they cover:
- password hashing, login, inactive users, duplicate users
- chunking, stemming, hybrid retrieval, role filtering, archived/new-version behaviour, PDF/DOCX/CSV search
- refusal of out-of-scope questions, offline extractive answers
- the full bundled question set (offline subset)
- question parsing, Bloom tagging, CO mapping, coverage, PO weights, repeat detection
- rubric validation and clamping of AI marks, diagram rules (flowchart, ER)
- the LLM code paths with a **mocked** model (grounded prompt, NOT_FOUND handling, error fallback, JSON retry)
- booting the real Streamlit app headlessly, logging in as each role and opening **every module**

What they do **not** cover: real Gemma 4 / Gemini API responses (no key in CI). After adding your key, do the live smoke check below once.

### Live smoke check (2 minutes, needs your key)
1. Admin, **Usage, Audit & Health**, click **Test LLM call** (expect `OK: pong`) and **Test embeddings** (expect a vector length).
2. Student, ask the attendance question: the answer should cite `[1]` and show sources; mode becomes `llm`.
3. Student, Snap-and-Explain with `circuit_rc.png`.
4. Faculty, Answer-Script Feedback with `Q3_good.png`, then `Q3_poor.png`; the good one must score clearly higher.
5. Diagram Reviewer with `flowchart_broken.png`: it must report a missing End.

---

## 9. Evaluate RAG quality

```bash
python scripts/eval_rag.py                    # uses your key if configured
LLM_BACKEND=none python scripts/eval_rag.py   # offline keyword-only (Windows: set LLM_BACKEND=none first)
```
It prints PASS/FAIL per question and a score. Add your own rows to `rag_test_questions.csv` (columns: `question, role, expected_keyword, should_answer, requires_embeddings`).

**Tuning** (`config/university.yaml`):
- Answers refused too often: lower `min_similarity` (try 0.50) or `min_term_coverage`.
- Wrong answers to out-of-scope questions: raise them.
- Long documents answered poorly: lower `chunk_chars` (try 700) and re-upload.

---

## 10. Deploy to Streamlit Community Cloud (live URL)

Organizers want a public link. Streamlit Community Cloud is free and builds straight from GitHub. **You must do these steps with your own GitHub and Streamlit accounts.**

### 10.1 Put the code on GitHub
1. Create an account at <https://github.com> and a **new public repository** (for example `campus-ai`). Do not add a README or license there (they already exist here).
2. In the project folder:
```bash
git init
git add .
git commit -m "Campus AI Platform"
git branch -M main
git remote add origin https://github.com/YOUR-USER/campus-ai.git
git push -u origin main
```
3. **Check that no secret was committed:** open the repo on GitHub; `.env` and `.streamlit/secrets.toml` must NOT be there (they are in `.gitignore`). If a key leaked, delete it in AI Studio and create a new one.

### 10.2 Create the Streamlit app
1. Go to <https://share.streamlit.io> and sign in with GitHub (allow access to your repo).
2. Click **Create app**, then **Deploy a public app from GitHub**.
3. Fill in: **Repository** = `YOUR-USER/campus-ai`, **Branch** = `main`, **Main file path** = `app.py`.
4. Optionally set a custom **App URL** (for example `campus-ai-vitap`).
5. Open **Advanced settings**:
   - **Python version:** 3.11
   - **Secrets:** paste
     ```toml
     GEMINI_API_KEY = "your-key"
     ```
     (add `DATABASE_URL` too if you want persistence, section 11)
6. Click **Deploy**. The first build takes a few minutes. When it finishes you get a URL like `https://campus-ai-vitap.streamlit.app`.

### 10.3 After deployment
- Open the URL in a **private window**, log in as each demo role, and run the live smoke check (section 8).
- **Share the URL** with organizers. Put it in your README, your submission form and your slides.
- To change a secret later: app menu, **Settings**, **Secrets**, save (the app restarts).
- Every `git push` to `main` redeploys automatically.

### 10.4 Before the demo (important)
- **Free apps sleep after a period of inactivity.** Visit the URL 10 to 15 minutes before judging so it is awake (if you see a "wake up" button, click it).
- **Without a database URL the data is temporary** on Streamlit Cloud (see section 11): after a restart, your uploaded documents disappear and the sample data is loaded again. For a stable demo, use Postgres (below) or re-upload your documents right before the demo.
- Keep a **backup screen recording** of the demo in case of a network or quota problem.
- The demo accounts and passwords are public in the README. That is fine for a hackathon; for a real deployment set `show_demo_credentials: false` and change the passwords (User Management).

---

## 11. Database options

| Option | Setup | Persistent on Streamlit Cloud? | Best for |
|---|---|---|---|
| **SQLite** (default) | none; file `data/campus.db` | **No** (resets on restart/redeploy) | local use, quick demos |
| **Neon Postgres** (free tier) | 3 minutes | **Yes** | live demo with uploaded data surviving restarts |
| **Supabase Postgres** (free tier) | 3 minutes | Yes | alternative to Neon |

The app uses SQLAlchemy, so only the connection string changes. Tables are created automatically.

### Neon setup (recommended for the live URL)
1. Sign up at <https://neon.tech> and create a project.
2. Copy the **connection string** (looks like `postgresql://user:pass@ep-xxxx.region.aws.neon.tech/neondb?sslmode=require`).
3. Local: add `DATABASE_URL=...` to `.env`. Cloud: add `DATABASE_URL = "..."` in **Secrets**.
4. Restart. Check **Usage, Audit & Health**: `Database` should say `postgresql`.
5. The first start seeds demo users and sample documents into this database (once, because it only seeds an empty database).

### Supabase setup
Project Settings, Database, copy the **connection pooling (Session mode)** URI and use it as `DATABASE_URL`. Replace the `[YOUR-PASSWORD]` placeholder.

**Notes:** vectors are stored as JSON text and searched in Python, which is fine for thousands of chunks. For very large corpora, move to pgvector.

---

## 12. Local models with Ollama (no API)

Use this for the privacy story: nothing leaves your machine.
1. Install Ollama from <https://ollama.com>.
2. Pull models (confirm the exact Gemma 4 tag with `ollama list` / the Ollama model library):
```bash
ollama pull gemma4
ollama pull nomic-embed-text
```
3. In `.env`:
```
LLM_BACKEND=ollama
OLLAMA_MODEL=gemma4
OLLAMA_EMBED_MODEL=nomic-embed-text
OLLAMA_URL=http://localhost:11434
```
4. Restart the app and use **Test LLM call** in the Admin health page.

Streamlit Community Cloud cannot reach an Ollama server on your laptop, so Ollama is for local runs. If you change embedding models, re-index each document (Knowledge Base Manager, **Re-index**). Telugu quality depends on the model you pick.

---

## 13. Docker

```bash
cp .env.example .env        # add GEMINI_API_KEY
docker compose up --build
```
Open <http://localhost:8501>. The `data/` folder is mounted so the SQLite database survives restarts. To use Ollama from inside Docker, set `OLLAMA_URL=http://host.docker.internal:11434`.

---

## 14. Configuration reference

**Environment variables / Streamlit secrets** (env var wins over secrets):

| Name | Default | Meaning |
|---|---|---|
| `GEMINI_API_KEY` | none | enables Gemma 4 via the Gemini API |
| `LLM_BACKEND` | `gemini` if a key exists, else `none` | `gemini`, `ollama` or `none` |
| `GEMMA_MODEL` | `gemma-4-26b-a4b-it` | text/vision model |
| `EMBED_MODEL` | `gemini-embedding-001` | embedding model |
| `DATABASE_URL` | `sqlite:///data/campus.db` | database connection string |
| `OLLAMA_URL`, `OLLAMA_MODEL`, `OLLAMA_EMBED_MODEL` | see `.env.example` | local backend |

**`config/university.yaml`:** `name`, `tagline`, `icon`, `show_demo_credentials`, `seed_sample_data`, `not_found_message`, and `rag.*` (`top_k`, `chunk_chars`, `chunk_overlap`, `min_similarity`, `min_term_coverage`, `suggested_questions`).

**Rubric file** (`config/rubrics/*.yaml`): `course`, then `questions: [{id, text, max_marks, points: [{id, text, marks}]}]`. Point marks must add up to `max_marks`.

**Outcomes file** (`config/outcomes/*.yaml`): `cos` (id, text), `pos`, `matrix` (CO to PO levels 1-3), `bloom_targets` (`max_pct` / `min_pct` per Bloom level).

**PO weight formula:** for each PO, sum over COs of (CO share of marks in percent x mapping level / 3).

---

## 15. Adapting it for another university

1. Edit `config/university.yaml` (name, tagline, thresholds). Set `show_demo_credentials: false`.
2. Replace the files in `config/rubrics/` and `config/outcomes/` with your department's.
3. Set `seed_sample_data: false`, then upload your real documents in the Knowledge Base Manager.
4. Create real users (User Management, or bulk CSV) and change the admin password.
5. Deploy (sections 10 to 13).

---

## 16. Adding a new module

1. Create `modules/my_feature.py`:
```python
import streamlit as st

MANIFEST = {"id": "my_feature", "label": "My Feature", "icon": "🧪",
            "roles": ["student", "faculty"], "order": 45}

def render(user):
    st.header("My Feature")
    st.write(f"Hello {user['name']}")
```
2. Restart the app. The module appears in the menu for those roles only (`core/registry.py` scans the folder). Use `core.llm.generate / generate_json` for the model, `core.db` for storage and `core.audit.log` for the audit trail.

---

## 17. Security and privacy notes

- Passwords are hashed with salted PBKDF2-SHA256. There is no SSO yet; add OAuth/SAML for production.
- Role checks happen server-side in code: menus are built from the role and retrieval filters by audience in the database query.
- **Using the Gemini API sends the prompt, retrieved passages and images to Google.** Do not upload real student scripts or personal data to a public demo. Read Google's current terms for your key type (free-tier data handling differs from paid). For sensitive data, use Ollama locally.
- Uploaded document text is treated as untrusted data and fenced in the prompt, and the model is told to ignore instructions inside it. This reduces, but does not eliminate, prompt injection risk.
- AI marks are suggestions only, clamped to the rubric and saved only after faculty approval; every AI action is in the audit log.
- Streamlit Cloud: never put keys in code; use Secrets. The sample accounts are public, so do not store real data on a demo deployment.

---

## 18. Troubleshooting

| Symptom | Likely cause and fix |
|---|---|
| `ModuleNotFoundError` | Virtual environment not active, or `pip install -r requirements.txt` not run. |
| `streamlit: command not found` | Activate the venv, or run `python -m streamlit run app.py`. |
| Port already in use | `streamlit run app.py --server.port 8502` |
| Login page keeps rejecting | Use the demo credentials exactly; if you changed passwords or the DB, run `python scripts/reset_db.py`. |
| Banner says "No LLM key configured" | `.env` / secrets missing or key empty. Restart after editing `.env`. |
| `404 model not found` | Model ID changed. Set `GEMMA_MODEL` (section 4). |
| `429` / `RESOURCE_EXHAUSTED` | Free-tier rate limit. Wait, or reduce parallel use, or enable billing. |
| Answers always "not found" | Lower `min_similarity` / `min_term_coverage`; check the Knowledge Base page shows chunks and `embedded` > 0; use **Test a search**. |
| Telugu questions not found | Telugu needs embeddings (key set) and a multilingual embedding model; keyword-only mode cannot match Telugu to English text. |
| `embedded` is 0 for every document | Embedding call failed. Admin health page, **Test embeddings** shows the error. Then press **Re-index**. |
| PDF upload: "No selectable text" | Scanned PDF. Export pages as images and upload those (needs the key for OCR). |
| Uploaded documents vanished on Streamlit Cloud | SQLite is temporary there. Use `DATABASE_URL` (section 11). |
| `psycopg2` / SSL error with Postgres | Keep `?sslmode=require` in the URL; use the pooled/session URL for Supabase. |
| Feedback module says "Evaluation failed: ... valid JSON" | The model returned prose. Click again (the app already retries once), or switch `GEMMA_MODEL`. |
| App slow on first load | Cold start plus seeding embeddings; later loads are fast (cached). |
| Camera does not work | Browsers require HTTPS or localhost for the camera. The Cloud URL is HTTPS. |

---

## 19. Known limitations

- Live Gemma 4 / Gemini calls are **not covered by the automated tests** (they need a key); only the mocked paths are. Do the live smoke check once after setting your key.
- Handwriting accuracy depends on image quality. Low-confidence rubric points are flagged for manual review, but marks are never final until a teacher approves.
- Vector search is in-process (NumPy), not a vector database. Fine for demo and department scale.
- No SSO, no email, no per-document permissions beyond the audience field.
- Scanned PDFs must be uploaded as images for OCR.
- The CO and Bloom tags from the offline rule-based mode are heuristic; AI tags are better but still need faculty review.
- Sample documents are fictional and not VIT-AP policy.

---

## 20. 3-minute demo script

1. **Problem (20 s):** information is scattered, evaluation paperwork is heavy; universities need a reusable open-source tool.
2. **Admin (40 s):** log in as `admin`, upload a PDF in Knowledge Base Manager, show chunks and embeddings, run **Test a search**.
3. **Student (60 s):** ask the attendance question (cited answer), then an out-of-scope question (honest "not found"), then Snap-and-Explain with `circuit_rc.png`.
4. **Faculty (40 s):** Answer-Script Feedback with `Q3_good.png` vs `Q3_poor.png`; then CO-PO Mapper on the sample paper showing repeats and the Excel download.
5. **Close (20 s):** "One open-source platform, role-based, config-driven. Any university deploys it with one command and customises it by editing YAML. It runs on the Gemini API or fully local with open weights."

---

## 21. Hackathon submission checklist

- [ ] Public GitHub repo, Apache-2.0 license present
- [ ] Live URL (Streamlit Cloud) opens in a private window and demo logins work
- [ ] `GEMINI_API_KEY` set in Cloud Secrets; **Test LLM call** and **Test embeddings** pass
- [ ] `DATABASE_URL` (Neon) set if you want uploaded data to persist
- [ ] `python -m pytest -q` shows 44 passed
- [ ] Live smoke check done (section 8)
- [ ] No secrets in the repo (`.env`, `secrets.toml` absent on GitHub)
- [ ] README top section shows the live URL
- [ ] Backup demo video recorded
- [ ] Short note in your submission explaining how Gemma 4 is used (text, vision, embeddings via the Gemini API, local mode through Ollama)

License: Apache-2.0.
