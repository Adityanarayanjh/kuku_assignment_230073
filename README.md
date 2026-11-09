# PromoForge - promo discovery for microdrama screenplays

Reads a show's screenplay files, finds 20-60 s promo clips (hook in the first 3 s, unresolved cliffhanger,
understandable to a new viewer), validates every timestamp against the source, ranks them with a diversity
penalty, and gives an editor a review UI. The editor stays the final decision-maker.

> **Confidential data:** the Kuku screenplays must never be committed or published. `.gitignore` already excludes
> `data/`, `outputs/`, `submission/`, `cache/`, `.env` and `*.zip`. Tests use synthetic screenplays only.

## How it works (5 stages)
| Stage | File | AI or code? |
|---|---|---|
| A. Parse screenplay -> episodes/scenes/events with timestamps in seconds, flag malformed data | `src/parser.py` | code |
| B. Find candidate moments (rule peaks + optional LLM proposals) | `src/candidate_extractor.py`, `src/lexicon.py` | rules (+LLM) |
| C. Score hook / cliffhanger / clarity / emotion / coherence / promo potential with evidence quotes | `src/evaluator.py` | rules **or** LLM |
| D/E. Choose exact start/end (never mid-line, 20-60 s); try a short "context bridge" for confusing openings | `src/boundary_optimizer.py` | code |
| F. Independent validation, accept/review/reject gates, diversity-aware ranking | `validator.py`, `ranker.py`, `diversity.py` | code |

Final score = 0.20 hook + 0.20 cliffhanger + 0.20 clarity + 0.15 emotion + 0.15 coherence + 0.10 promo potential,
with **hard gates**: hook < 6 or cliffhanger < 6 => rejected; either < 7 => overall capped at 7.5. Everything is in `config.py`.

## Install (Windows PowerShell)
```powershell
cd promoforge
py -3.11 -m venv .venv          # any Python 3.10+ works
.\.venv\Scripts\Activate.ps1    # if blocked: Set-ExecutionPolicy -Scope Process Bypass
pip install -r requirements.txt
```

## Run on a show (offline engine - free, no key)
Put the screenplay `.zip` (or the folder of `.txt` files) under `data\input\`, then:
```powershell
python main.py run --input "data\input\I_Married_a_Broken_Soldier__He_was_a_Duke.zip" --engine heuristic --out outputs\run_heuristic
python verify_outputs.py --input "data\input\I_Married_a_Broken_Soldier__He_was_a_Duke.zip" --run outputs\run_heuristic
streamlit run app.py
```
A **new show** needs no code changes: point `--input` at its folder/zip, or upload files on the Dashboard page.
Files must follow the same text layout (`EPISODE NUMBER`, `SCENE nn`, `**SPEAKER (tone):`, `Start: (mm:ss.cc) | End: (...)`).
Different layouts: add/adjust a parser in `src/parser.py`; the rest of the pipeline only sees `Episode/Scene/Event`.

## Run with an LLM (Gemini)
```powershell
copy .env.example .env          # then put your key in .env (GEMINI_API_KEY=...)
python main.py estimate --input "data\input\<show>.zip"      # token / call estimate first
python main.py run --input "data\input\<show>.zip" --engine llm --out outputs\run_llm
```
Model/provider are env vars (`LLM_MODEL`, `LLM_PROVIDER`). To add another provider, subclass `LLMClient` in
`src/llm_client.py` and implement `_call()`. JSON validation, retries, 429 back-off and an on-disk cache are inherited.
Raw model replies are saved untouched in `outputs\run_llm\raw_llm_responses\`.
**Status of LLM mode:** unit-tested with a scripted client; it has not been run against a live model in the build environment.

## Tests
```powershell
python -m unittest discover -s tests -p "test_*.py"
```

## Output files (`outputs\<run>\`)
`parsed_episodes.json` (stage A) | `candidates.json` (seeds + windows) | `evaluations_raw.json` | `ranked_promos.json/.csv` |
`review_queue.json` | `rejected_candidates.json` | `validation_report.json` | `evaluation_report.json` | `best_per_episode.json` |
`editorial_report.md` | `run_manifest.json` | `run_log.txt`. Every excerpt line carries `event_id` and source line numbers.

## Deploy for free
Streamlit Community Cloud needs a GitHub repo. Because the screenplay is confidential, deploy only the code (no data) and
upload files at run time - or keep it private and share a screen recording. Never put the zip or `outputs/` in the repo.

## Known limits
Screenplay-only (no visuals/audio); script timings are not frame-accurate; offline scoring is keyword-based and
over-rewards `?`/shouting; no prediction of views or conversion (needs real performance data after posting).
