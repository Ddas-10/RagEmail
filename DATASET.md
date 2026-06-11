# DATASET.md

## Source

**Enron Email Dataset**
- Kaggle: https://www.kaggle.com/datasets/wcukierski/enron-email-dataset
- License: Public domain (released by FERC during Enron investigation)
- Original size: ~517,401 emails from 150 users (single emails.csv, ~1.4 GB)

## Slice Selection

### Criteria
- Mailboxes selected: `skilling-j` (4,139 emails), `lay-k` (5,937), `dasovich-j` (28,234)
- Date window: **October 2000 – March 2001** (active period; California energy crisis + succession planning)
- Thread filter: top-20 threads by message count (min 3 messages per thread)

### Final counts

| Metric | Count |
|---|---|
| Threads | 20 |
| Messages | ~501 |
| Attachments (synthetic TXT for demo) | 2 |
| Approximate indexed text | ~2–4 MB |
| Total index chunks (estimated after ingest) | ~600–900 |

### Thread topics in the slice

| Thread | Messages | Topic |
|---|---|---|
| teams | 60 | Enron management team updates |
| succession plan | 56 | Executive succession planning (Lay → Skilling) |
| study? | 26 | Policy / regulatory study requests |
| congratulations | 26 | Internal employee congratulations |
| year end 2000 performance feedback | 26 | Performance review cycle |
| letter proposal and more | 24 | Business proposals |
| genesis park | 24 | Enron venture / real estate |
| message points | 24 | Internal communication strategy |
| draft of materials for governor | 24 | California energy crisis / Governor Gray Davis |
| panel on valuation | 23 | Analyst and investor panel |
| california (brief) | 22 | California energy market briefing |
| california | 22 | Energy regulation and pricing |
| directions | 22 | Meeting/travel logistics |
| energy issues | 22 | Regulatory energy issues |
| thanks | 20 | Internal correspondence |
| cal-iso wants pwr generators to forward bid | 20 | California ISO market operations |
| follow up with alpert's office | 20 | Government relations |
| cpuc inquiry re gas customer turnbacks | 20 | CPUC regulatory inquiry |
| ypo panel, april 24, 2001 | 20 | Young Presidents Organisation panel |

## Preprocessing

1. Decoded all `quoted-printable` and `base64` encoded bodies (Python `email` stdlib)
2. Normalised subjects for thread grouping (stripped Re:/Fwd:/AW: prefixes)
3. Deduplicated by Message-ID (same message in multiple mailboxes counted once)
4. Thread detection: 2-pass In-Reply-To / References graph → root hash → `thread_id = T-<sha1[:8]>`
5. Attachments: Kaggle CSV contains email bodies only (no binary files). Two synthetic `.txt`
   attachments added for demo purposes (California energy brief, Succession plan draft).

## How to recreate

```powershell
# 1. Download from Kaggle (requires Kaggle account + API key)
kaggle datasets download wcukierski/enron-email-dataset -p data/raw/
Expand-Archive data\raw\enron-email-dataset.zip -DestinationPath data\raw\enron\

# 2. Extract slice (PowerShell — use backtick ` for line continuation)
python scripts/extract_slice.py `
  --csv data/raw/enron/emails.csv `
  --mailboxes skilling-j lay-k dasovich-j `
  --start 2000-10-01 `
  --end 2001-03-31 `
  --min-thread-size 3 `
  --output data/slice/

# 3. Build index
python ingest.py --data-dir data/slice/
```
