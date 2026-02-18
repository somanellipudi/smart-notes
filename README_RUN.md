# Install and Run

## Prerequisites
- Python 3.10+ recommended
- pip

## Setup
```bash
python -m venv .venv
```

Windows:
```bash
.venv\Scripts\activate
```

macOS/Linux:
```bash
source .venv/bin/activate
```

Install dependencies:
```bash
pip install -r requirements.txt
```

## Run the app
```bash
streamlit run app.py
```

## Optional environment variables
Create a `.env` file if needed. Common options:
- `OPENAI_API_KEY`
- `LLM_MODEL`
- `ENABLE_VERIFIABLE_MODE`
