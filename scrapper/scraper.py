"""
coches.net stealth scraper
--------------------------
Scrapes car listings from coches.net for each brand/model/year
combination listed in combinations.csv.

Setup:
    pip install -r requirements.txt
    patchright install chromium
    cp .env.example .env   # then edit OUTPUT_DIR

Run:
    python scraper.py

Resume after interruption:
    Just run it again — progress.json tracks completed combinations
    and already-written CSVs are kept.

Anti-bot notes:
    coches.net is protected by a DataDome-style JavaScript challenge that
    fingerprints the browser and scores every request. To avoid detection
    the scraper:
      - Uses **patchright** (a Playwright fork with built-in stealth patches
        that hide navigator.webdriver, fix the WebGL vendor, restore
        chrome.runtime, etc.) instead of vanilla playwright.
      - Layers **playwright-stealth** on top to patch any remaining signals.
      - Defaults to HEADED mode (HEADLESS=false). Headless Chromium has
        additional fingerprintable tells; on a server, run inside Xvfb.
      - Uses ONE browser context per combination so the challenge cookie
        set on page 1 persists across all subsequent pages.
      - Navigates to subsequent pages by CLICKING the "next page" link
        rather than jumping directly to the paginated URL.
      - Uses generous, randomised delays between pages and combinations.
"""

import asyncio
import csv
import json
import os
import random
import re
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
# patchright is a drop-in replacement for playwright with built-in stealth
# patches (hides navigator.webdriver, fixes WebGL vendor, restores chrome.runtime,
# disables CDP detection, etc.). Its API is identical to playwright.async_api.
from patchright.async_api import async_playwright, TimeoutError as PlaywrightTimeoutError
from playwright_stealth import Stealth

load_dotenv()

# ── Configuration (all overridable via .env) ──────────────────────────────────
OUTPUT_DIR        = Path(os.getenv("OUTPUT_DIR",        "./output"))
COMBINATIONS_FILE = Path(os.getenv("COMBINATIONS_FILE", "combinations.csv"))
PROGRESS_FILE     = Path(os.getenv("PROGRESS_FILE",     "progress.json"))
MIN_DELAY         = float(os.getenv("MIN_DELAY",        "5.0"))   # seconds between pages
MAX_DELAY         = float(os.getenv("MAX_DELAY",        "10.0"))  # seconds between pages
COMBO_DELAY_MIN   = float(os.getenv("COMBO_DELAY_MIN",  "12.0"))  # seconds between combinations
COMBO_DELAY_MAX   = float(os.getenv("COMBO_DELAY_MAX",  "20.0"))
RATE_LIMIT_WAIT   = float(os.getenv("RATE_LIMIT_WAIT",  "90.0"))  # wait when blocked
MAX_RETRIES       = int(os.getenv("MAX_RETRIES",        "3"))
HEADLESS          = os.getenv("HEADLESS", "false").lower() == "true"

# Single Stealth instance reused across all contexts. It applies the
# evasions via init scripts, so creating it once is enough.
_STEALTH = Stealth()

# ── URL slug mappings ─────────────────────────────────────────────────────────
BRAND_MAP = {
    "mercedesbenz": "mercedes-benz",
    "landrover":    "land-rover",
}

MODEL_MAP = {
    "c5 aircross":       "c5-aircross",
    "land cruiser":      "land-cruiser",
    "santa fe":          "santa-fe",
    "xtrail":            "x-trail",
    "crv":               "cr-v",
    "classe glc":        "clase-glc",
    "range rover":       "range-rover",
    "range rover velar": "range-rover-velar",
    "range rover evoque":"range-rover-evoque",
    "corolla verso":     "corolla-verso",
    "space star":        "space-star",
    "mazda 3":           "mazda3",
    "cmax":              "c-max",
}

CSV_COLUMNS = [
    "brand", "model", "year_filter",
    "year_circulation", "km", "hp",
    "fuel_type", "price_eur", "title", "url",
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def build_url(brand: str, model: str, year: int, page: int = 1) -> str:
    b = BRAND_MAP.get(brand.lower(), brand.lower())
    m = MODEL_MAP.get(model.lower(), model.lower().replace(" ", "-"))
    url = f"https://www.coches.net/segunda-mano/{b}/{m}/?MinYear={year}&MaxYear={year}"
    if page > 1:
        url += f"&pagina={page}"
    return url


def combo_key(brand: str, model: str, year: int) -> str:
    return f"{brand}|{model}|{year}"


def combo_filename(brand: str, model: str, year: int) -> str:
    safe_model = model.lower().replace(" ", "_").replace("/", "_")
    safe_brand = brand.lower().replace(" ", "_")
    return f"{safe_brand}_{safe_model}_{year}.csv"


def load_combinations() -> list[tuple]:
    combos = []
    with open(COMBINATIONS_FILE, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            combos.append((
                row["brand"].strip(),
                row["model"].strip(),
                int(row["year"].strip()),
            ))
    return combos


def load_progress() -> dict:
    if PROGRESS_FILE.exists():
        try:
            return json.loads(PROGRESS_FILE.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
    return {}


def save_progress(progress: dict) -> None:
    PROGRESS_FILE.write_text(json.dumps(progress, indent=2, ensure_ascii=False), encoding="utf-8")


def log(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# ── JS extractor (run inside the browser page) ────────────────────────────────

JS_EXTRACT = """
() => {
    const scripts = document.querySelectorAll('script:not([src])');
    for (const s of scripts) {
        const m = s.textContent.match(
            /window\\.__INITIAL_PROPS__\\s*=\\s*JSON\\.parse\\(("[^"\\\\]*(?:\\\\.[^"\\\\]*)*")\\)/s
        );
        if (m) {
            try {
                const parsed = JSON.parse(JSON.parse(m[1]));
                const res = parsed.initialResults;
                if (res) return {
                    items:        res.items        || [],
                    totalResults: res.totalResults || 0,
                    totalPages:   res.totalPages   || 1,
                };
            } catch(e) {}
        }
    }
    return null;
}
"""


async def extract_page_data(page) -> dict | None:
    """Extract __INITIAL_PROPS__ from the currently loaded page."""
    try:
        await page.wait_for_timeout(2_000)   # let any lazy JS finish
        return await page.evaluate(JS_EXTRACT)
    except Exception as e:
        log(f"  ⚠  Extraction error: {e}")
        return None


async def click_next_page(page) -> bool:
    """
    Click the confirmed 'next page' link on coches.net.

    The site uses a single <a aria-label="Página siguiente"> anchor. Clicking
    it navigates to the ID-based URL format the site uses internally:
      /segunda-mano/?MakeIds[0]=X&ModelIds[0]=Y&...&pg=N
    That is intentional — we let the site handle its own URL routing rather
    than constructing paginated URLs ourselves.

    Returns True if the click succeeded and the page navigated.
    """
    sel = 'a[aria-label="Página siguiente"]'
    try:
        el = await page.query_selector(sel)
        if el is None:
            log("  ⚠  Next-page link not found in DOM")
            return False
        if not await el.is_visible():
            log("  ⚠  Next-page link is not visible")
            return False
        disabled = await el.get_attribute("aria-disabled")
        if disabled == "true":
            log("  ⚠  Next-page link is disabled (already on last page?)")
            return False

        old_url = page.url
        await el.click()
        await page.wait_for_function(
            f"() => location.href !== {json.dumps(old_url)}",
            timeout=10_000,
        )
        await page.wait_for_load_state("domcontentloaded", timeout=20_000)
        return True
    except Exception as e:
        log(f"  ⚠  click_next_page error: {e}")
        return False


# ── Per-combination scraper ───────────────────────────────────────────────────

async def scrape_combination(
    browser, brand: str, model: str, year: int
) -> list[dict]:
    """
    Return a deduplicated list of listing dicts for one brand/model/year.

    Uses a single browser context throughout so that the anti-bot challenge
    cookie (set when page 1 loads) is preserved for all subsequent pages.
    """
    context = await browser.new_context(
        user_agent=(
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        locale="es-ES",
        viewport={"width": 1280, "height": 900},
    )
    # Layer playwright-stealth on top of patchright's built-in patches.
    # This injects init scripts that further mask automation signals
    # (navigator.permissions, plugins, languages, sec-ch-ua, etc.).
    await _STEALTH.apply_stealth_async(context)
    page = await context.new_page()

    all_items: list[dict] = []
    seen_ids: set[str] = set()

    try:
        # ── Page 1 ───────────────────────────────────────────────────────────
        url_p1 = build_url(brand, model, year, 1)
        data = None
        for attempt in range(1, MAX_RETRIES + 1):
            log(f"  → page 1 (attempt {attempt}): {url_p1}")
            try:
                await page.goto(url_p1, wait_until="domcontentloaded", timeout=30_000)
            except PlaywrightTimeoutError:
                log(f"  ⏱  Timeout loading page 1")
                if attempt < MAX_RETRIES:
                    await asyncio.sleep(RATE_LIMIT_WAIT)
                continue

            data = await extract_page_data(page)
            if data and data.get("items"):
                break
            if attempt < MAX_RETRIES:
                log(f"  ⏳ No data on page 1 — challenge not passed? Waiting {RATE_LIMIT_WAIT}s …")
                await asyncio.sleep(RATE_LIMIT_WAIT)

        if not data or not data.get("items"):
            log(f"  ✗ No listings found for {brand}/{model}/{year}")
            return []

        total_results = data["totalResults"]
        total_pages   = data["totalPages"]
        log(f"  ✓ {total_results} listings across {total_pages} page(s)")

        for item in data["items"]:
            if item["id"] not in seen_ids:
                seen_ids.add(item["id"])
                all_items.append(item)

        # ── Pages 2+ — navigate by clicking "Página siguiente" ──────────────
        for p in range(2, total_pages + 1):
            delay = random.uniform(MIN_DELAY, MAX_DELAY)
            log(f"  ⏳ Waiting {delay:.1f}s …")
            await asyncio.sleep(delay)

            page_data = None
            for attempt in range(1, MAX_RETRIES + 1):
                log(f"  → page {p} (attempt {attempt})")

                if attempt == 1:
                    # First attempt: click the next-page link from wherever we are.
                    # The site transitions to its internal ID-based URL automatically.
                    clicked = await click_next_page(page)
                    if not clicked:
                        log(f"  ⚠  Next-page click failed on page {p}, skipping")
                        break
                else:
                    # Retry: we're already on the correct URL, just reload it.
                    log(f"  ↺  Reloading page {p} …")
                    try:
                        await page.reload(wait_until="domcontentloaded", timeout=30_000)
                    except PlaywrightTimeoutError:
                        log(f"  ⏱  Timeout reloading page {p}")
                        if attempt < MAX_RETRIES:
                            await asyncio.sleep(RATE_LIMIT_WAIT)
                        continue

                page_data = await extract_page_data(page)
                if page_data and page_data.get("items"):
                    break
                if attempt < MAX_RETRIES:
                    log(f"  ⏳ No data on page {p} — waiting {RATE_LIMIT_WAIT}s …")
                    await asyncio.sleep(RATE_LIMIT_WAIT)

            if page_data and page_data.get("items"):
                for item in page_data["items"]:
                    if item["id"] not in seen_ids:
                        seen_ids.add(item["id"])
                        all_items.append(item)
            else:
                log(f"  ⚠ Could not retrieve page {p}, continuing …")

    finally:
        try:
            await context.close()
        except Exception:
            pass

    return all_items


# ── CSV writer ────────────────────────────────────────────────────────────────

def save_csv(brand: str, model: str, year: int, items: list[dict]) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    filepath = OUTPUT_DIR / combo_filename(brand, model, year)

    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_COLUMNS)
        for item in items:
            writer.writerow([
                brand,
                model,
                year,
                item.get("year", ""),
                item.get("km", ""),
                item.get("hp", ""),
                item.get("fuelType", ""),
                item.get("price", ""),
                item.get("title", "").replace("\n", " ").strip(),
                "https://www.coches.net" + item.get("url", ""),
            ])

    return filepath


# ── Main ──────────────────────────────────────────────────────────────────────

async def main() -> None:
    combinations = load_combinations()
    progress     = load_progress()

    total     = len(combinations)
    completed = sum(1 for v in progress.values() if v.get("done"))
    log(f"Loaded {total} combinations — {completed} already done, {total - completed} remaining")

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=HEADLESS)

        async def ensure_browser():
            nonlocal browser
            if not browser.is_connected():
                log("  ⚠ Browser died — relaunching …")
                try:
                    await browser.close()
                except Exception:
                    pass
                browser = await pw.chromium.launch(headless=HEADLESS)

        try:
            for i, (brand, model, year) in enumerate(combinations, 1):
                key = combo_key(brand, model, year)

                if progress.get(key, {}).get("done"):
                    log(f"[{i}/{total}] ✓ skip  {brand} / {model} / {year}")
                    continue

                await ensure_browser()
                log(f"[{i}/{total}] ▶ {brand} / {model} / {year}")

                try:
                    items = await scrape_combination(browser, brand, model, year)
                    filepath = save_csv(brand, model, year, items)
                    progress[key] = {
                        "done":      True,
                        "rows":      len(items),
                        "file":      str(filepath),
                        "timestamp": datetime.now().isoformat(),
                    }
                    save_progress(progress)
                    log(f"  ✅ {len(items)} unique rows → {filepath.name}")

                except Exception as e:
                    log(f"  ❌ Error: {e}")
                    progress[key] = {
                        "done":      False,
                        "error":     str(e),
                        "timestamp": datetime.now().isoformat(),
                    }
                    save_progress(progress)

                if i < total:
                    delay = random.uniform(COMBO_DELAY_MIN, COMBO_DELAY_MAX)
                    log(f"  ⏳ Waiting {delay:.1f}s before next combination …\n")
                    await asyncio.sleep(delay)

        finally:
            try:
                await browser.close()
            except Exception:
                pass

    # ── Summary ──────────────────────────────────────────────────────────────
    done   = sum(1 for v in progress.values() if v.get("done"))
    errors = sum(1 for v in progress.values() if not v.get("done"))
    log(f"\n{'='*50}")
    log(f"Done: {done}  |  Errors / no-data: {errors}  |  Total: {total}")
    if errors:
        log("Combinations with errors (re-runnable):")
        for k, v in progress.items():
            if not v.get("done"):
                log(f"  {k} — {v.get('error', 'no data found')}")


if __name__ == "__main__":
    asyncio.run(main())
