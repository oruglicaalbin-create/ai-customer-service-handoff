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

## Data

- **BANKING77** by PolyAI (Casanueva et al., 2020, "Efficient Intent Detection with Dual Sentence
  Encoders"), licensed [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). 13,083 banking
  customer queries labelled with 77 intents. Downloaded automatically from
  [PolyAI-LDN/task-specific-datasets](https://github.com/PolyAI-LDN/task-specific-datasets) by
  `src/load_data.py`; not redistributed in this repo.

## Folder layout

| Folder     | Contents                                                        |
|------------|-----------------------------------------------------------------|
| `src/`     | Python code                                                     |
| `data/`    | Simulated bank systems (fake data); `data/raw/` = downloads, not committed |
| `results/` | Metrics and plots produced by the evaluation                    |
| `cache/`   | Saved API responses so re-runs cost nothing (not committed)     |
