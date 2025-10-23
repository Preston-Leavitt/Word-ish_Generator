# Word‑ish Generator

A simple Flask web app that invents a fake word and uses the OpenAI API to generate a plausible dictionary‑style definition. It also shows a random image and plays a random song from the `static` folder.

## Prerequisites

- Python 3.9+ (tested with 3.13)
- An OpenAI API key

## Setup (Windows PowerShell)

You can install just the minimal dependencies:

```powershell
# From the project root
python -m pip install --upgrade pip ; \
python -m pip install Flask openai
```

Or install everything from requirements (heavier and slower, includes ML libs no longer needed for runtime):

```powershell
python -m pip install -r requirements.txt
```

Set your OpenAI API key (temporary for the session):

```powershell
$env:OPENAI_API_KEY = "<your_api_key_here>"
```

## Run the app

```powershell
python .\generator.py
```

The server binds to localhost. Open your browser to:

- http://127.0.0.1:5000/

## How it works

- The app constructs a fake word using letter probabilities from `place.csv` and `relative.csv`.
- It calls the OpenAI Chat Completions API to generate a short, family‑friendly definition.
- A random image and song are selected from `static/images` and `static/songs`.

## Configuration

- Port: set the `PORT` environment variable if you need a different port than 5000.
- API key: set `OPENAI_API_KEY` in your environment. If it’s not set, the app will still run and show a placeholder message for the definition.

## Notes

- The previous local ML model (torch/transformers) has been removed from the runtime path. Files like `mlgenerator.py` remain for reference but are not used by the web app.
- Audio files in `static/songs` should be compatible with your browser.
