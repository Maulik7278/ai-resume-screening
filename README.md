# AI Resume Screening

Batch-processes a folder of PDF resumes, extracts Python and AI/LLM/Agentic
evidence, applies a deterministic hard eligibility gate, enriches with
optional GitHub profile data, scores and ranks eligible candidates, and
writes an auditable `results.json`.

## How it works

```
Input folder (PDFs)
      │
      ▼
 parse PDF (pypdf)
      │
      ▼
 deterministic regex extraction  (always runs — no API key required)
      │
      ▼
 Cohere semantic analysis (best-effort; failure isolated per resume)
      │
      ▼
 merge evidence (deterministic ∪ Cohere)
      │
      ▼
 hard eligibility gate: Python evidence AND AI/LLM/Agentic evidence
      │  (deterministic — Cohere informs it, never decides it alone)
      ▼
 GitHub enrichment (best-effort, only if a GitHub URL was found)
      │
      ▼
 scoring (Python/AI evidence + project quality + GitHub activity)
      │
      ▼
 rank (eligible first, by total score) → results.json
```

Every stage that touches an external system (Cohere, GitHub, PDF parsing)
is wrapped so a failure on one resume is recorded on that resume only —
the rest of the batch always continues. The pipeline works even with **no
Cohere API key at all**: it falls back to deterministic regex extraction,
and `llm_status` is reported as `"not_available"` rather than faked.

## Project layout

```
ai-resume-screening/
├── src/
│   ├── config.py        # env-var driven settings
│   ├── models.py         # Pydantic models (Cohere schema + result schema)
│   ├── parser.py          # PDF → text, resume discovery
│   ├── extractor.py       # deterministic regex/keyword extraction
│   ├── eligibility.py      # deterministic hard eligibility gate
│   ├── scorer.py           # scoring for eligible candidates
│   ├── github.py           # GitHub profile enrichment (cached)
│   ├── pipeline.py          # batch orchestration, per-resume isolation
│   └── llm/
│       └── cohere_provider.py  # the ONLY module that talks to Cohere
├── tests/                # pytest unit tests (all external calls mocked)
│   └── fixtures/          # small synthetic PDF resumes used by tests
├── main.py                 # CLI entry point
├── requirements.txt
├── .env.example
└── .gitignore
```

## Setup

```bash
pip install -r requirements.txt
# then edit .env and set COHERE_API_KEY (and optionally GITHUB_TOKEN)
```

`GITHUB_TOKEN` is optional — GitHub enrichment still works unauthenticated,
just at a much lower rate limit (60 requests/hour vs. 5000/hour).

## Running

```bash
python main.py --input ./resumes --output ./output/results.json
```

Options:
- `--concurrency N` — max concurrent resumes processed at once (default
  from `MAX_WORKERS` in `.env`, falls back to 5).
- `--verbose` — debug-level logging.

The input directory can contain any number of PDFs with any filenames —
nothing is hard-coded to a naming pattern or candidate list.

## Testing

Two levels, per the project spec:

**Level 1 — automated unit tests** (no API keys required, all external
calls mocked):

```bash
pytest
```

Covers eligibility rules, scoring (including the shallow-wrapper penalty
and evidence saturation), PDF parser failures (missing/corrupt/unreadable
files), Cohere failure isolation, GitHub failure isolation, batch-level
per-resume isolation, and ranking order.

**Level 2 — real batch test.** Once real resumes are available, drop them
into `resumes/` and run:

```bash
python main.py --input ./resumes --output ./output/results.json
```

Then check `output/results.json`:
- `summary.total_discovered` matches the number of PDFs in the folder
- every PDF appears exactly once in `candidates`
- `eligible` candidates have a non-null `score_breakdown`
- non-eligible (but successfully parsed) candidates have `eligibility_reasons`
- failed-to-parse candidates have `parse_status: "failed"` and an entry in `errors`
- `rank` is ascending and `score_breakdown.total_score` is non-increasing
  down the list (within the eligible group, and separately within the
  rejected group)

## A note on the demo/fixture PDFs

`tests/fixtures/*.pdf` are small synthetic resumes generated for automated
testing only (e.g. `eligible_resume.pdf`, `python_only_resume.pdf`,
`blank_resume.pdf` for a scanned-with-no-text-layer case). They are **not**
real candidate data. No `results.json` claiming to represent the real
50-candidate batch is included in this repository — that file is generated
fresh by running the CLI against the real `resumes/` folder once it's
supplied, per the project's "do not fabricate results" requirement.

## Design notes / trade-offs

- **Concurrency**: `ThreadPoolExecutor` with a small, configurable worker
  count (`MAX_WORKERS`, `COHERE_CONCURRENCY`, `GITHUB_CONCURRENCY` in
  `.env`). Bounded on purpose — no uncontrolled parallel fan-out.
- **Rate limiting**: Cohere Trial keys are capped at 40 API calls/minute.
  A shared sliding-window rate limiter (`COHERE_CALLS_PER_MINUTE`, default
  35) throttles every worker thread so the batch stays under that cap
  instead of hitting repeated `429` errors. On a genuine 429, the retry
  backoff is extended (20s) rather than the normal short backoff, since
  retrying immediately after a rate-limit hit just gets rate-limited again.
- **Retries**: Cohere calls retry up to `COHERE_MAX_RETRIES` times (default
  2) with small bounded backoff (not exponential-forever). Malformed JSON
  output and network/timeout errors are both retried once, then the
  failure is recorded and the batch moves on.
- **Candidate name resolution**: preferred order is Cohere's extracted
  name → a deterministic heuristic that scans the top of the resume text
  for a name-like line → the PDF filename, only as an absolute last
  resort. This means a candidate still gets their real name in the output
  even if Cohere fails or isn't configured for that resume.
- **Strengths/concerns/summary fallback**: when Cohere doesn't produce an
  analysis for a resume (not configured, or failed after retries), these
  three fields are populated from a deterministic narrative built off the
  same evidence used for eligibility and scoring — instead of being left
  silently empty. The fallback is always paired with an entry in `errors`
  explaining why the LLM step didn't run, so it's clear the narrative is
  keyword-based rather than a semantic read of the resume.
- **Caching**: an in-memory cache keyed by resume-text hash avoids
  re-calling Cohere if identical resume content is encountered twice in a
  run; GitHub responses are cached in-memory per username. No database —
  a single run's cache is enough for this scope.
- **Eligibility stays deterministic**: the hard Python+AI gate is decided
  from evidence lists (deterministic regex ∪ validated Cohere evidence),
  never from an LLM's free-form yes/no judgement. This keeps the decision
  predictable, testable without an API key, and auditable in the output
  JSON via `eligibility_reasons`.
