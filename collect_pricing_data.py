#!/usr/bin/env python3
"""Web scraping tool that extracts product pricing data using Playwright and LLMs."""
import asyncio
import json
import re
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple

import os
from openai import OpenAI

from bs4 import BeautifulSoup
from pydantic import BaseModel, ValidationError
from playwright.async_api import (
    Error as PWError, async_playwright, Page, ElementHandle,
)

# -----------------------------
# Config
# -----------------------------
URL = "https://www.mediamarkt.be/fr/"

# -----------------------------
# Pydantic schema for extraction
# -----------------------------
class Product(BaseModel):
    """A single product with name, model and price."""
    name: Optional[str] = None
    model: Optional[str] = None
    #sku: Optional[str] = None
    price: Optional[float] = None


class ProductsPayload(BaseModel):
    """Container for a list of Product items."""
    products: List[Product] = []


# -----------------------------
# LLM Client abstraction
# -----------------------------
class LLMClient:  # pylint: disable=too-few-public-methods
    """Base class for LLM clients."""
    async def generate_json(self, system: str, user: str) -> Dict[str, Any]:
        """Generate a JSON response from the LLM."""
        raise NotImplementedError

class OpenAIClient(LLMClient):  # pylint: disable=too-few-public-methods
    """
    Uses OpenAI python SDK.
    Env:
      - OPENAI_API_KEY
      - OPENAI_MODEL (optional, default: gpt-4o-mini)
    """
    def __init__(self):
        # lecture centralisée de la config
        self.api_key = os.environ.get("OPENAI_API_KEY")
        self.base_url = os.environ.get(
            "OPENAI_BASE_URL", "http://localhost:8080/v1"
        )
        self.cloud_model = os.environ.get("OPENAI_MODEL",
                                          "gpt-4o-mini")
        self.local_model = os.environ.get("LOCAL_MODEL",
                                          "Qwen3-14B-Q4_K_M.gguf")

        # ----- MODE LOCAL LLAMA.CPP -----
        if not self.api_key:
            print("⚠️ OPENAI_API_KEY not set → using local llama.cpp server")

            self.client = OpenAI(
                base_url=self.base_url or "http://localhost:8080/v1",
                api_key="not-needed"
            )

            self.model = self.local_model
            self.mode = "local"

        # ----- MODE OPENAI CLOUD -----
        else:
            print("🌐 Using OpenAI cloud API")
            self.client = OpenAI(api_key=self.api_key)
            self.model = self.cloud_model
            self.mode = "cloud"


    async def generate_json(self, system: str, user: str) -> Dict[str, Any]:
        def _call():
            print(f"[LLM:{self.mode}] model={self.model}")

            approx_tokens = (len(system) + len(user)) // 4
            if self.mode == "local" and approx_tokens > 6500:
                raise RuntimeError(
                    f"Prompt too large for local ctx: "
                    f"~{approx_tokens} tokens (ctx=8192). "
                    f"Reduce candidates/HTML or increase "
                    f"llama.cpp -c."
                )

            resp = self.client.chat.completions.create(
                model=self.model,
                temperature=0,
                max_tokens=1200,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                response_format={"type": "json_object"},
            )

            content = resp.choices[0].message.content or "{}"

            # parse JSON robustly
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                # save bad output for debugging
                with open("llm_bad_output.txt", "w",
                          encoding="utf-8") as f:
                    f.write(content)
                print("Saved llm_bad_output.txt")

                blob = _extract_json_blob(content)
                if blob:
                    try:
                        return json.loads(blob)
                    except json.JSONDecodeError:
                        # ask model to repair the JSON once
                        repair_system = (
                            "You are a JSON repair tool. "
                            "Return ONLY valid JSON. "
                            "No markdown, no commentary."
                        )
                        repair_user = (
                            "Fix the following JSON so that "
                            "it is valid JSON. "
                            "Do not change meaning, only fix "
                            "syntax/escaping.\n\n"
                            f"{blob}"
                        )
                        repaired = self.client.chat.completions.create(
                            model=self.model,
                            temperature=0,
                            messages=[
                                {"role": "system",
                                 "content": repair_system},
                                {"role": "user",
                                 "content": repair_user},
                            ],
                            response_format={
                                "type": "json_object"
                            },
                        ).choices[0].message.content or "{}"

                        return json.loads(repaired)

                raise RuntimeError(
                    f"Model did not return JSON. "
                    f"First 500 chars:\n{content[:500]}"
                ) from None

        return await asyncio.to_thread(_call)

# -----------------------------
# Utilities
# -----------------------------

def _css_escape(s: str) -> str:
    # Escape minimal for attribute selectors
    return s.replace("\\", "\\\\").replace('"', '\\"')

def _extract_json_blob(text: str) -> str | None:
    start = text.find("{")
    end = text.rfind("}") + 1
    if 0 <= start < end:
        return text[start:end]
    return None

def _compact_text(text: str, max_chars: int = 14000) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > max_chars:
        return text[:max_chars] + " ..."
    return text

async def _safe_inner_text(el: ElementHandle) -> str:
    try:
        txt = await el.inner_text()
        return _compact_text(txt, 300)
    except (PWError, OSError):
        return ""

async def _element_descriptor(
    el: ElementHandle,
) -> Dict[str, Any]:
    attrs = await el.evaluate("""e => ({
        id: e.id || null,
        name: e.getAttribute('name'),
        type: e.getAttribute('type'),
        role: e.getAttribute('role'),
        placeholder: e.getAttribute('placeholder'),
        ariaLabel: e.getAttribute('aria-label'),
        className: e.className || null
    })""")

    # selector "stable" : priorité id, puis name, puis aria-label
    selector = None
    if attrs.get("id"):
        selector = f"#{attrs['id']}"
    elif attrs.get("name"):
        selector = f'input[name="{attrs["name"]}"]'
    elif attrs.get("ariaLabel"):
        # attention aux guillemets
        al = attrs["ariaLabel"].replace('"', '\\"')
        selector = f'input[aria-label="{al}"]'

    txt = await _safe_inner_text(el)
    return {"attrs": attrs, "text": txt, "selector": selector}


async def _build_stable_selector(
    el: ElementHandle,
    tag_hint: str | None = None,
) -> Optional[str]:
    """
    Try to build a stable selector for an element.
    Priority:
      1) #id
      2) tag[name="..."]
      3) tag[aria-label="..."]
      4) tag[placeholder="..."]
      5) XPath (fallback)
    """
    try:
        attrs = await el.evaluate("""e => ({
            tag: e.tagName.toLowerCase(),
            id: e.id || null,
            name: e.getAttribute('name'),
            aria: e.getAttribute('aria-label'),
            placeholder: e.getAttribute('placeholder')
        })""")
    except (PWError, OSError):
        return None

    tag = (tag_hint or attrs.get("tag") or "*").lower()

    if attrs.get("id"):
        return f"#{_css_escape(attrs['id'])}"

    if attrs.get("name"):
        return f'{tag}[name="{_css_escape(attrs["name"])}"]'

    if attrs.get("aria"):
        sel = _css_escape(attrs["aria"])
        return f'{tag}[aria-label="{sel}"]'

    if attrs.get("placeholder"):
        sel = _css_escape(attrs["placeholder"])
        return f'{tag}[placeholder="{sel}"]'

    # XPath fallback (less pretty, but works often)
    try:
        xpath = await el.evaluate(
            """e => {
                const idx = (sib, name) => {
                  if (!sib) return 1;
                  let i = 1;
                  for (let s = sib.previousElementSibling; s;
                       s = s.previousElementSibling) {
                    if (s.tagName === name) i++;
                  }
                  return i;
                };
                let path = '';
                for (; e && e.nodeType === 1; e = e.parentElement) {
                  const name = e.tagName.toLowerCase();
                  const i = idx(e, e.tagName);
                  path = '/' + name + '[' + i + ']' + path;
                }
                return path ? 'xpath=' + path : null;
            }"""
        )
        return xpath
    except (PWError, OSError):
        return None


# -----------------------------
# AgentQL-like wrapper
# -----------------------------
@dataclass
class SmartPage:
    """Wraps a Playwright Page with LLM-powered element finding and data extraction."""
    page: Page
    llm: LLMClient

    async def query_data(
        self,
        query: str,
        max_chars: int = 18000,
    ) -> Dict[str, Any]:
        """
        Extracts structured data from the current page.
        The `query` is treated as a schema hint (GraphQL-like).
        """
        html = await self.page.content()
        soup = BeautifulSoup(html, "html.parser")

        # Reduce noise: remove script/style
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()

        text = _compact_text(
            soup.get_text(" ", strip=True), max_chars
        )

        system = (
            "You are a web extraction engine.\n"
            "You will receive a schema-like query and page text.\n"
            "Return STRICT JSON matching the schema intent.\n"
            "If a field is missing, use null.\n"
            "Do NOT add commentary.\n"
        )

        user_payload = {
            "query": query,
            "page_text": text,
            "instructions": [
                "Extract at most 10 products from this page",
                "Prices must be JSON numbers.",
                "Normalize price format: remove currency "
                "symbols and spaces.",
                "Thousands separators must be removed.",
                "Decimal separator must be a dot.",
                "Example conversions: "
                "€1.299,95 -> 1299.95 | "
                "1,299.95 € -> 1299.95",
            ],
            "output_format": "Return JSON object (not markdown).",
        }

        return await self.llm.generate_json(
            system=system, user=json.dumps(user_payload)
        )



    async def get_by_prompt(
        self,
        prompt: str,
        selectors: str = (
            "input, textarea, button, select, a, "
            "[role='searchbox']"
        ),
        max_candidates: int = 80,
    ) -> Optional[str]:
        """
        AgentQL-like element finder.
        Returns a *selector string* (CSS or xpath=...)
        instead of ElementHandle to avoid stale handles.
        """
        # Ensure page is ready enough
        try:
            await self.page.wait_for_load_state(
                "domcontentloaded"
            )
        except (PWError, OSError):
            pass

        elements = await self.page.query_selector_all(selectors)
        if not elements:
            return None

        candidates: List[Tuple[str, Dict[str, Any]]] = []

        # Build a compact candidate list (visible only)
        for el in elements:
            try:
                if not await el.is_visible():
                    continue
            except (PWError, OSError):
                continue

            selector = await _build_stable_selector(el)
            if not selector:
                continue

            try:
                desc = await el.evaluate("""e => ({
                    tag: e.tagName.toLowerCase(),
                    text: (e.innerText || '').slice(0, 80),
                    type: e.getAttribute('type'),
                    role: e.getAttribute('role'),
                    placeholder: e.getAttribute('placeholder'),
                    ariaLabel: e.getAttribute('aria-label'),
                    name: e.getAttribute('name'),
                    id: e.id || null
                })""")
            except (PWError, OSError):
                desc = {"tag": None}

            # Keep payload small
            desc_compact = {
                "selector": selector,
                "tag": desc.get("tag"),
                "id": desc.get("id"),
                "name": desc.get("name"),
                "ariaLabel": desc.get("ariaLabel"),
                "placeholder": desc.get("placeholder"),
                "type": desc.get("type"),
                "role": desc.get("role"),
                "text": (desc.get("text") or ""),
            }

            candidates.append((selector, desc_compact))
            if len(candidates) >= max_candidates:
                break

        if not candidates:
            return None

        system = (
            "You are a DOM element selector.\n"
            "Given a target description and a list of "
            "candidates, choose the best match.\n"
            'Return STRICT JSON only: '
            '{"index": <int>, "confidence": <0..1>}.\n'
            "If none match, return index=-1.\n"
        )

        payload = {
            "target": prompt,
            "candidates": [c[1] for c in candidates],
            "rules": [
                "Prefer inputs/textareas for typing queries.",
                "Prefer searchbox role when available.",
                "If target mentions 'next page', prefer "
                "pagination controls (Next, Suivant, ›).",
            ],
        }

        resp = await self.llm.generate_json(
            system=system, user=json.dumps(payload)
        )
        idx = int(resp.get("index", -1))
        conf = float(resp.get("confidence", 0.0))

        if idx < 0 or idx >= len(candidates) or conf < 0.35:
            return None

        return candidates[idx][0]

    # --- helper: click by prompt ---
    async def click_by_prompt(
        self,
        prompt: str,
        selectors: str = "input, textarea, button, select, a",
        max_candidates: int = 60,
    ) -> bool:
        """Click an element found by LLM prompt matching."""
        el = await self.get_by_prompt(
            prompt, selectors=selectors,
            max_candidates=max_candidates,
        )
        if not el:
            return False
        try:
            await el.click()
            return True
        except (PWError, OSError):
            return False


async def safe_fill(
    page: Page, el: ElementHandle,
    text: str, retries: int = 3,
) -> None:
    """Fill an element with text, retrying on transient errors."""
    last_exc = None
    for _ in range(retries):
        try:
            await el.wait_for_element_state(
                "visible", timeout=5000
            )
            await el.fill(text)
            return
        except PWError as e:
            last_exc = e
            if "not attached to the DOM" in str(e):
                break
            await page.wait_for_timeout(250)
    raise last_exc or RuntimeError("safe_fill failed")

async def safe_click(
    page: Page, el: ElementHandle, retries: int = 3,
) -> None:
    """Click an element, retrying on transient errors."""
    last_exc = None
    for _ in range(retries):
        try:
            await el.wait_for_element_state(
                "visible", timeout=5000
            )
            await el.click()
            return
        except PWError as e:
            last_exc = e
            if "not attached to the DOM" in str(e):
                break
            await page.wait_for_timeout(250)
    raise last_exc or RuntimeError("safe_click failed")

# -----------------------------
# Your business logic (same as your original flow)
# -----------------------------
async def _do_extract_pricing_data(
    sp: SmartPage,
) -> List[Dict[str, Any]]:
    """Extract product pricing data from the current page."""
    query = """
    {
        products[] {
            name
            model
            price
        }
    }"""
    raw = await sp.query_data(query)

    # Validate & normalize via Pydantic
    try:
        payload = ProductsPayload.model_validate(raw)
        # Convert to dict
        out: List[Dict[str, Any]] = []
        for p in payload.products:
            out.append(p.model_dump())
        return out
    except ValidationError:
        # If the LLM didn't return the exact shape, best-effort fallback:
        products = raw.get("products", [])
        if isinstance(products, list):
            return products
        return []


async def _search_product(
    sp: SmartPage, product: str,
    min_price: int, max_price: int,
) -> bool:
    """Search for a product on the current page."""
    _ = min_price, max_price  # reserved for future filtering
    page = sp.page
    await page.wait_for_load_state("domcontentloaded")
    await page.wait_for_timeout(800)

    # (A) Fallbacks simples (pas de LLM, pas brouillon)
    for sel in [
        "#twotabsearchtextbox",
        "input[name='field-keywords']",
        "input[type='search']",
        "input[role='searchbox']",
        "input[name*='search' i]",
        "input[id*='search' i]",
    ]:
        el = await page.query_selector(sel)
        if el:
            try:
                await safe_fill(page, el, product)
                await el.press("Enter")
                await page.wait_for_timeout(1500)
                return True
            except (PWError, OSError):
                pass

    # (B) LLM: trouver l'input
    search_input = await sp.get_by_prompt(
        "the search input field where you can type "
        "a product name"
    )
    if not search_input:
        print("Search input field not found.")
        return False

    # (C) Fill avec gestion detach
    try:
        await safe_fill(page, search_input, product)
        await search_input.press("Enter")
    except (PWError, OSError) as e:
        if "not attached to the DOM" in str(e):
            search_input = await sp.get_by_prompt(
                "the search input field where you can "
                "type a product name"
            )
            if not search_input:
                return False
            await safe_fill(page, search_input, product)
            await search_input.press("Enter")
        else:
            raise

    await page.wait_for_timeout(1500)
    return True

async def _go_to_the_next_page(sp: SmartPage) -> bool:
    """Navigate to the next page of results."""
    page = sp.page

    # 1) BestBuy (souvent)
    for sel in [
        "a[aria-label='Next']",
        "a[aria-label*='Next']",
        "button[aria-label*='Next']",
        "a:has-text('Next')",
        "button:has-text('Next')",
        "a:has-text('Suivant')",
        "button:has-text('Suivant')",
        "a[rel='next']",
    ]:
        el = await page.query_selector(sel)
        if el and await el.is_visible():
            try:
                await el.click()
                await page.wait_for_timeout(1200)
                return True
            except (PWError, OSError):
                pass

    # 2) seulement si rien trouvé → LLM mais très réduit
    return await sp.click_by_prompt(
        "the next page button in pagination (Next)",
        selectors="a,button",
        max_candidates=25,
    )

async def extract_pricing_data(
    sp: SmartPage,
    product: str,
    min_price: int,
    max_price: int,
    max_pages: int = 3,
) -> list:
    """Extract pricing data across multiple pages."""
    print(
        f"Searching for product: {product} "
        f"with price range: {min_price} - {max_price}"
    )
    if await _search_product(
        sp, product, min_price, max_price
    ) is False:
        print("Failed to search for the product.")
        return []

    current_page = 1
    pricing_data = []
    while current_page <= max_pages:
        print(
            f"Extracting pricing data on page "
            f"{current_page}..."
        )
        pricing_data_on_page = await _do_extract_pricing_data(
            sp
        )
        print(f"{len(pricing_data_on_page)} products found")

        pricing_data.extend(pricing_data_on_page)

        if not await _go_to_the_next_page(sp):
            print("No more next page.")
            break

        current_page += 1

    return pricing_data


async def main():
    """Entry point: search and extract pricing data."""
    llm = OpenAIClient()

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=False)
        page = await browser.new_page()
        await page.goto(URL)

        sp = SmartPage(page=page, llm=llm)

        pricing_data = await extract_pricing_data(
            sp,
            product="gpu nvidia",
            min_price=50,
            max_price=2800,
            max_pages=3,
        )

        print(json.dumps(pricing_data[:10], indent=2))
        await browser.close()


if __name__ == "__main__":
    asyncio.run(main())
