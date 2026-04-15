# coches.net — Technical Scraping Guide

This document explains exactly how coches.net is structured, why standard HTTP
requests fail, and how the Playwright scraper in `scraper.py` works around that.
It is written so that any developer (or coding assistant) can understand the
approach and extend or debug it.

---

## 1. Why plain `requests` / `httpx` won't work

When you do a raw HTTP GET to a coches.net listing page, the server returns a
minimal **8 KB JavaScript shell** — not the actual listings. Example of what
you get:

```html
<!DOCTYPE html>
<html>
  <head>
    <meta charset="UTF-8">
    <noscript><title>Ups! Parece que algo no va bien...</title></noscript>
    <meta name="viewport" content="...">
    ...
  </head>
  <body>
    <div id="app"></div>
    <script src="https://s.ccdn.es/main.99227c4b.js"></script>
  </body>
</html>
```

The listings are NOT in this HTML. They are injected into `window.__INITIAL_PROPS__`
by the server **only when it detects a real browser navigation** (it checks the
`Sec-Fetch-Mode: navigate` header, which browsers set automatically but which
cannot be spoofed from `requests` or `fetch()` calls due to browser security
policies).

**Solution:** Use Playwright to launch a real Chromium browser. The server sees
a genuine navigation request and returns the full server-rendered HTML.

---

## 2. Where the listing data lives

When the page is loaded by a real browser, the server embeds all listing data
for that page directly into the HTML inside an inline `<script>` tag:

```html
<script>
  window.__INITIAL_PROPS__ = JSON.parse("{\"breadcrumb\":[...],\"initialResults\":{...}}");
</script>
```

The value is a **JSON string that has been `JSON.stringify`-ed and then
embedded as a JS string literal** — so you need to `JSON.parse` it twice:

```python
# Playwright page.evaluate equivalent:
import json, re

html = page.content()   # full HTML from Playwright

match = re.search(
    r'window\.__INITIAL_PROPS__\s*=\s*JSON\.parse\((".*?")\)',
    html, re.DOTALL
)
data = json.loads(json.loads(match.group(1)))
results = data["initialResults"]
```

---

## 3. Structure of `initialResults`

```json
{
  "totalResults": 106,
  "totalPages":   4,
  "items": [ ...35 items per page... ],
  "aggregations": { ... }
}
```

| Field          | Type | Meaning                                              |
|----------------|------|------------------------------------------------------|
| `totalResults` | int  | Total listings matching the filter                   |
| `totalPages`   | int  | Number of pages (35 items per page)                  |
| `items`        | list | The listings for this page (see below)               |

---

## 4. Structure of a single listing item

Real example extracted from `toyota/rav4/2020`:

```json
{
  "id":           "61070455",
  "make":         "TOYOTA",
  "model":        "Rav4",
  "year":         2020,
  "km":           178500,
  "hp":           218,
  "fuelType":     "Híbrido",
  "fuelTypeId":   4,
  "price":        23790,
  "title":        "TOYOTA Rav4 2.5l 220H Advance",
  "url":          "/toyota-rav4-2.5l-220h-advance-5p-electrico-/-hibrido-2020-en-jaen-61070455-covo.aspx",

  "location": {
    "mainProvince":   "Jaén",
    "mainProvinceId": 23,
    "regionLiteral":  "Andalucía",
    "cityLiteral":    "Linares"
  },

  "bodyTypeId":        6,
  "offerType":         { "literal": "Ocasión" },
  "isProfessional":    true,
  "hasWarranty":       true,
  "hasReservation":    true,
  "environmentalLabel":"ECO",
  "photos":            ["https://a.ccdn.es/...jpg", "..."],
  "img":               "https://a.ccdn.es/...jpg",

  "creationDate":      "2025-07-19T13:02:36Z",
  "publicationDate":   "2025-07-19"
}
```

Fields we extract to CSV:

| CSV column         | Source field   | Notes                                         |
|--------------------|----------------|-----------------------------------------------|
| `brand`            | *(from query)* | Taken from combinations.csv, not the listing  |
| `model`            | *(from query)* | Same                                          |
| `year_filter`      | *(from query)* | The year we searched for                      |
| `year_circulation` | `year`         | The car's actual registration year            |
| `km`               | `km`           | Odometer in km                                |
| `hp`               | `hp`           | Horsepower (CV)                               |
| `fuel_type`        | `fuelType`     | e.g. "Diésel", "Gasolina", "Híbrido"          |
| `price_eur`        | `price`        | Listed price in EUR                           |
| `title`            | `title`        | Full listing title string                     |
| `url`              | `url`          | Prepend `https://www.coches.net` to get full URL |

---

## 5. URL structure

### Listing page (slug-based)

```
https://www.coches.net/segunda-mano/{brand}/{model}/?MinYear={year}&MaxYear={year}
https://www.coches.net/segunda-mano/{brand}/{model}/?MinYear={year}&MaxYear={year}&pagina={page}
```

Examples:
```
https://www.coches.net/segunda-mano/toyota/rav4/?MinYear=2020&MaxYear=2020
https://www.coches.net/segunda-mano/toyota/rav4/?MinYear=2020&MaxYear=2020&pagina=2
https://www.coches.net/segunda-mano/kia/stonic/?MinYear=2021&MaxYear=2021
https://www.coches.net/segunda-mano/mercedes-benz/clase-glc/?MinYear=2019&MaxYear=2019
https://www.coches.net/segunda-mano/land-rover/range-rover/?MinYear=2023&MaxYear=2023
```

> ⚠️ **Year filter note:** The parameters are `MinYear` and `MaxYear` — NOT
> `aniodesde`/`aniohasta`. The latter appear in some older URLs but are ignored
> by the server.

> ⚠️ **Pagination note:** Page 1 has no `&pagina=` parameter. Page 2 onwards
> adds `&pagina=2`, `&pagina=3`, etc.

### Name mappings (CSV → URL slug)

Some brand/model names in the combinations CSV differ from coches.net URL slugs:

**Brands:**
| CSV value      | URL slug       |
|----------------|----------------|
| mercedesbenz   | mercedes-benz  |
| landrover      | land-rover     |

**Models:**
| CSV value          | URL slug           |
|--------------------|--------------------|
| c5 aircross        | c5-aircross        |
| land cruiser       | land-cruiser       |
| santa fe           | santa-fe           |
| xtrail             | x-trail            |
| crv                | cr-v               |
| classe glc         | clase-glc          |
| range rover        | range-rover        |
| range rover velar  | range-rover-velar  |
| range rover evoque | range-rover-evoque |
| corolla verso      | corolla-verso      |
| space star         | space-star         |
| mazda 3            | mazda3             |
| cmax               | c-max              |

All other models: replace spaces with hyphens (`model.replace(" ", "-")`).

### Individual listing URL
```
https://www.coches.net{item["url"]}
```
e.g. `https://www.coches.net/toyota-rav4-2.5l-220h-advance-5p-...-61070455-covo.aspx`

---

## 6. JavaScript extraction snippet (used by the scraper)

The scraper injects this into the browser with `page.evaluate()`:

```javascript
() => {
    const scripts = document.querySelectorAll('script:not([src])');
    for (const s of scripts) {
        const m = s.textContent.match(
            /window\.__INITIAL_PROPS__\s*=\s*JSON\.parse\(("[^"\\]*(?:\\.[^"\\]*)*")\)/s
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
```

Returns `null` when the page returned the bare shell (rate-limited).
The scraper treats `null` as a signal to wait and retry.

---

## 7. Rate-limiting behaviour

coches.net rate-limits after roughly 6–10 rapid requests in quick succession.
When rate-limited, it returns the 8 KB shell (no `__INITIAL_PROPS__`) instead
of the full page. The scraper detects this by checking whether the extracted
data is `null` or has 0 items, then waits `RATE_LIMIT_WAIT` seconds (default
60 s) before retrying.

**Recommended delays** (set in `.env`):
- Between pages within a combination: 3.5 – 7 s (randomised)
- Between combinations: 10 – 18 s (randomised)
- On rate-limit detection: 60 s flat before retry

---

## 8. Resume / fault-tolerance design

`progress.json` is written after **every completed combination**, even if the
script is interrupted. Format:

```json
{
  "toyota|rav4|2020": {
    "done":      true,
    "rows":      106,
    "file":      "output/toyota_rav4_2020.csv",
    "timestamp": "2024-06-01T14:32:10"
  },
  "kia|stonic|2021": {
    "done":      false,
    "error":     "Timeout",
    "timestamp": "2024-06-01T14:35:02"
  }
}
```

On restart, the scraper skips any combination where `"done": true`.
Combinations with `"done": false` (errors) will be retried automatically.

To **force a full re-scrape**, simply delete `progress.json`.

---

## 9. Output files

One CSV per combination, named: `{brand}_{model}_{year}.csv`

Examples:
```
output/
├── toyota_rav4_2020.csv
├── kia_stonic_2021.csv
├── ford_focus_2018.csv
├── mercedes-benz_clase-glc_2019.csv
└── ...
```

Each file has columns:
```
brand, model, year_filter, year_circulation, km, hp, fuel_type, price_eur, title, url
```

---

## 10. Setup & run

```bash
# 1. Install dependencies
pip install -r requirements.txt
playwright install chromium

# 2. Configure output location
cp .env.example .env
# Edit OUTPUT_DIR in .env to your desired path

# 3. Put your combinations list in place
cp /path/to/your/combinations.csv combinations.csv

# 4. Run
python scraper.py

# 5. To resume after interruption, just run again:
python scraper.py
```

---

## 11. Known edge cases

| Issue | Cause | Handling |
|-------|-------|----------|
| 0 results for a valid combination | Rate-limited | Auto-retry with 60 s wait |
| URL slug mismatch (wrong brand/model name) | Mapping incomplete | Add to `BRAND_MAP`/`MODEL_MAP` in `scraper.py` |
| `year_circulation` differs from `year_filter` | Site year filter is approximate | Filter in post-processing if needed |
| Duplicate listings across pages | Pagination overlap | Deduplicated by listing `id` |
| Some Citroen C5 Aircross results include other models | The slug `/c5-aircross/` may partially overlap | Post-filter on `title.contains("C5")` if needed |
