# Bank customer-service assistant with optimised human handoff

Work in progress. Full description, results and how to run come in later steps.

## Setup (Windows, PowerShell)

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env      # then put your Anthropic API key in .env
python src/check_setup.py
```

## Folder layout

| Folder     | Contents                                                        |
|------------|-----------------------------------------------------------------|
| `src/`     | Python code                                                     |
| `data/`    | Simulated bank systems (fake data); `data/raw/` = downloads, not committed |
| `results/` | Metrics and plots produced by the evaluation                    |
| `cache/`   | Saved API responses so re-runs cost nothing (not committed)     |
