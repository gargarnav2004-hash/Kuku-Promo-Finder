# AI Promo Finder

Reads microdrama screenplays and suggests about 30-second promo clips: each one opens on a hook, ends on a cliffhanger and should make sense to a new viewer. By default it gives 15 promos, at most 2 per episode. Promos that pass a strict **hook test** and **cliffhanger test** come first. If fewer than 15 pass, the rest are filled with the closest near-misses, clearly labelled **NEAR MISS** with what they failed (turn off with `fill_with_near_misses: false`). All rejected promos go to `outputs/rejected_promos.csv` with the reason. You get exact timestamps, the source lines behind each pick, a rank, and rule-check warnings. A human approves or rejects each one in the app.

It reads **screenplay text only**. It cannot judge acting, visuals or sound.

## Setup (once, about 5 minutes)

1. Install Python 3.10 or newer from https://www.python.org/downloads/
2. Open a terminal in this folder and install the libraries:
   ```bash
   pip install -r requirements.txt
   ```
3. Get a free Gemini API key at https://aistudio.google.com/apikey
4. Copy `.env.example` to a new file named `.env` and paste your key after `GEMINI_API_KEY=`
5. Put the show's screenplay folder (one `.txt` per episode) inside `data/`, for example
   `data/I Married a Broken Soldier. He was a Duke/`

## Launch the app (one command)

```bash
streamlit run app.py
```

A browser tab opens. It has three pages:
1. **Promo Finder**: pick episodes, click **Analyze**, then read the ranked promo cards.
2. **Human Review**: Approve or Reject each promo with a reason and a note. Decisions are saved to `reviews.csv`.
3. **Quality**: review counts, top rejection reasons, hook and cliffhanger gate pass/fail counts with the rejected list, rule-check results and a CSV download.

## Run all episodes from the terminal (optional)

```bash
python run.py
```

The results go to `outputs/`: `promos.json`, `promos.csv`, `promos.md`, `validation_report.md`, `run_log.txt` (with token usage), `run_summary.json`, `rejected_promos.csv`, `all_ranked_promos.json`, `episode_tags.json` and `all_candidates.json`.

Other examples:
```bash
python run.py --episodes 1-3
```
```bash
python run.py --show "data/My New Show" --out outputs_new_show
```

## Running it on a new show (no code changes)

Drop the new show's folder of `.txt` screenplays into `data/`. Then either pick it in the app's **Show** dropdown, or set `show_dir` in `config.yaml`, or pass `--show` to `run.py`. The screenplays need the same format: `**NAME (emotion):**` or `**ACTION:**` blocks, each followed by a `Start: (MM:SS.ss) | End: (MM:SS.ss)` line.

## Changing the AI model

All AI calls go through `llm.py`. Change the model in `config.yaml`:
- A different Gemini model: change `model:`.
- OpenAI, Groq, OpenRouter, or Kuku's own gateway: set `provider: openai_compatible`, set `base_url` and `model`, then set `api_key_env` to the name of the key variable in `.env`.

Answers are cached in `cache/`, so re-running the same episodes costs nothing.

## Files

| File | What it does |
|---|---|
| `parse.py` | Turns screenplays into timestamped beats (no AI) |
| `llm.py` | The only file that calls the AI: cache, retries, fallback models, token log |
| `pipeline.py` | Tag hooks/cliffhangers (AI) → build clips (code) → judge (AI) → rule checks (code) → write outputs |
| `run.py` | Command-line runner |
| `app.py` | Streamlit app |
| `config.yaml` | All settings: model, promo length (target 30 s), number of promos (15), gate thresholds (4/5), judge weights |
