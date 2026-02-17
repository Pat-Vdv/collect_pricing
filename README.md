# Collect Pricing Data

Web scraping tool that extracts product pricing data from e-commerce websites using Playwright and LLMs.

## How it works

1. Opens a target website (default: MediaMarkt Belgium) in a Chromium browser via Playwright
2. Finds the search bar using CSS selector fallbacks or LLM-based element detection
3. Searches for a product and extracts structured pricing data (name, model, price) from results
4. Paginates through multiple pages of results
5. Returns normalized JSON output validated with Pydantic

The tool uses a **SmartPage** abstraction that combines Playwright with an LLM to:
- **Find elements by prompt** — describes what you're looking for in natural language, and the LLM picks the best matching DOM element
- **Extract structured data** — sends page text to the LLM with a schema hint and gets back normalized JSON

## Requirements

- Python 3.10+
- Playwright (with Chromium)
- An OpenAI-compatible LLM (cloud or local)

## Installation

```bash
python -m venv venv
source venv/bin/activate
pip install playwright beautifulsoup4 pydantic openai
playwright install chromium
```

## Configuration

The LLM backend is configured via environment variables:

| Variable | Description | Default |
|---|---|---|
| `OPENAI_API_KEY` | OpenAI API key. If unset, uses local mode | *(none)* |
| `OPENAI_MODEL` | Model name for cloud mode | `gpt-4o-mini` |
| `OPENAI_BASE_URL` | Base URL for the API | `http://localhost:8080/v1` |
| `LOCAL_MODEL` | Model name for local mode | `Qwen3-14B-Q4_K_M.gguf` |

### Local mode (llama.cpp)

If `OPENAI_API_KEY` is not set, the tool connects to a local llama.cpp server at `http://localhost:8080/v1`.

### Cloud mode (OpenAI)

Set `OPENAI_API_KEY` to use the OpenAI API directly.

## Usage

```bash
python collect_pricing_data.py
```

By default, it searches for "gpu nvidia" on MediaMarkt Belgium with a price range of 50-2800 and scrapes up to 3 pages of results.

To customize, edit the `main()` function parameters:

```python
pricing_data = await extract_pricing_data(
    sp,
    product="gpu nvidia",
    min_price=50,
    max_price=2800,
    max_pages=3,
)
```

## Output

JSON array of products:

```json
[
  {
    "name": "NVIDIA GeForce RTX 4070",
    "model": "RTX 4070",
    "price": 599.99
  }
]
```

## Architecture

- **`LLMClient` / `OpenAIClient`** — LLM abstraction with JSON generation, automatic repair on malformed output
- **`SmartPage`** — Playwright page wrapper with LLM-powered element finding (`get_by_prompt`) and data extraction (`query_data`)
- **`safe_fill` / `safe_click`** — Retry helpers for transient DOM detachment errors
- **`extract_pricing_data`** — Main orchestration: search, extract, paginate
