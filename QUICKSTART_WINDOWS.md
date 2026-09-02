# Quick Start (Windows / PowerShell)

No virtual environment needed — these steps install everything directly.

## 1. Extract the ZIP and open PowerShell in that folder

Right-click the extracted `ai-resume-screening` folder → "Open in Terminal"
(or open PowerShell and `cd` into it manually).

Confirm you're in the right place:
```powershell
dir
```
You should see `main.py`, `requirements.txt`, `src`, `tests`.

## 2. Install dependencies

```powershell
pip install -r requirements.txt
```

## 3. Add your Cohere API key



Paste your key into the `COHERE_API_KEY=` line, save, close Notepad.

> **Using a Cohere Trial key?** It's capped at 40 API calls/minute. The
> pipeline automatically throttles itself to stay under this (see
> `COHERE_CALLS_PER_MINUTE` in `.env`), so a 50-resume batch will take a
> few minutes rather than running at full speed — that's expected, not a
> bug.

> **Model errors ("model was removed")?** `COHERE_MODEL` defaults to
> `command-r-08-2024`, a currently supported model. If you see a `404`
> error mentioning a model was deprecated/removed, check `.env` still has
> that value (or another current model from
> https://docs.cohere.com/docs/models) and hasn't been overwritten with
> an old alias like `command-r-plus`.

## 4. Run the tests (sanity check — no API key needed for this step)

```powershell
pytest
```
Expect: `33 passed`.

## 5. Add your resumes

```powershell
copy C:\path\to\your\resumes\*.pdf resumes\
```

## 6. Run the real batch

```powershell
python main.py --input .\resumes --output .\output\results.json --verbose
```

## 7. Check the results

```powershell
notepad output\results.json
```

The summary line printed in the terminal (`discovered=... eligible=...`)
should match your resume count.

---

That's it — every step above uses your regular Python install directly,

