<!-- \# Dataset: FakeNewsNet



\## Source

\- \*\*URL\*\*: https://github.com/KaiDMML/FakeNewsNet

\- \*\*Clone Date\*\*: 2025-10-18

\- \*\*SHA256\*\*: `\[check your FakeNewsNet\_SHA256.txt file]`

\- \*\*Archive Location\*\*: `data/raw/fakenewsnet/original/FakeNewsNet\_repo\_snapshot.zip`



\## Citation

Shu, K., Mahudeswaran, D., Wang, S., Lee, D., \& Liu, H. (2018). "FakeNewsNet: A Data Repository for News Content, Social Context, and Spatiotemporal Information for Studying Fake News on Social Media." arXiv preprint arXiv:1811.05210. \[GitHub Repository](https://github.com/KaiDMML/FakeNewsNet)



\## Files \& Structure

FakeNewsNet organized into two news sources with binary fake/real labels:



\*\*dataset/ folder with 4 CSV files:\*\*

\- `gossipcop\_fake.csv` — 5,323 fake celebrity/gossip news articles

\- `gossipcop\_real.csv` — 16,817 real celebrity/gossip news articles

\- `politifact\_fake.csv` — 432 fake political news articles

\- `politifact\_real.csv` — 624 real political news articles

\- \*\*Total articles: 23,196\*\*



\## Schema (Key Files)



\### CSV files (gossipcop\_fake.csv, gossipcop\_real.csv, politifact\_fake.csv, politifact\_real.csv)



Tab-separated format with these columns:



| Position | Column | Type | Notes |

|----------|--------|------|-------|

| 1 | id | str | Article identifier (e.g., "gossipcop-2493749932" or "politifact15014") |

| 2 | news\_url | str | Original article URL |

| 3 | title | str | Article headline |

| 4 | tweet\_ids | str | Tab-separated list of Twitter IDs associated with article |



\*\*Row counts by file:\*\*

\- gossipcop\_fake.csv: 5,323 rows

\- gossipcop\_real.csv: 16,817 rows

\- politifact\_fake.csv: 432 rows

\- politifact\_real.csv: 624 rows

\- \*\*Total: 23,196 articles\*\*



\## Labels (as provided)

Binary labels encoded in filename:

\- `fake` — Misinformation articles

\- `real` — Factual articles



\*\*Label distribution:\*\*

\- Fake (Gossipcop): 5,323 articles

\- Real (Gossipcop): 16,817 articles

\- Fake (PolitiFact): 432 articles

\- Real (PolitiFact): 624 articles



\## Notes/Quirks

\- \*\*Labels implicit in filename:\*\* Fake/Real determined by filename, not in CSV columns

\- \*\*Tweet ID mappings only:\*\* Contains tweet IDs associated with articles, NOT full tweet data or engagement metrics (require Twitter API)

\- \*\*No explicit label column:\*\* Label must be derived from filename

\- \*\*Two news sources:\*\* Gossipcop (celebrity) and PolitiFact (politics) with imbalanced distribution (Gossipcop dominates)

\- \*\*Class imbalance:\*\* Gossipcop heavily skewed toward real (76% real vs. 24% fake); PolitiFact more balanced (41% real vs. 59% fake)

\- \*\*URLs vary in format:\*\* Some with https://, some without; inconsistent URL formatting

\- \*\*Tweet ID lists are tab-separated:\*\* Each article has multiple tweet IDs separated by tabs in single column



\## License/Terms

FakeNewsNet dataset is provided for research use. Check GitHub repository README for specific terms and attribution requirements.



\## Reproduction Steps



\### PowerShell (Windows)

```powershell

git clone https://github.com/KaiDMML/FakeNewsNet.git temp\_fakenewsnet

mkdir data\\raw\\fakenewsnet\\original -Force

Compress-Archive -Path temp\_fakenewsnet\\\* -DestinationPath data\\raw\\fakenewsnet\\original\\FakeNewsNet\_repo\_snapshot.zip

certutil -hashfile "data\\raw\\fakenewsnet\\original\\FakeNewsNet\_repo\_snapshot.zip" SHA256 > data\\raw\\fakenewsnet\\original\\FakeNewsNet\_SHA256.txt

Remove-Item -Recurse -Force temp\_fakenewsnet

Expand-Archive -Path data\\raw\\fakenewsnet\\original\\FakeNewsNet\_repo\_snapshot.zip -DestinationPath data\\raw\\fakenewsnet

```



\### Bash/Linux/macOS

```bash

git clone https://github.com/KaiDMML/FakeNewsNet.git temp\_fakenewsnet

mkdir -p data/raw/fakenewsnet/original

zip -r data/raw/fakenewsnet/original/FakeNewsNet\_repo\_snapshot.zip temp\_fakenewsnet

sha256sum data/raw/fakenewsnet/original/FakeNewsNet\_repo\_snapshot.zip > data/raw/fakenewsnet/original/FakeNewsNet\_SHA256.txt

rm -rf temp\_fakenewsnet

unzip -d data/raw/fakenewsnet data/raw/fakenewsnet/original/FakeNewsNet\_repo\_snapshot.zip

```



\## Example Rows



\*\*From gossipcop\_fake.csv:\*\*

```

gossipcop-2493749932	www.dailymail.co.uk/tvshowbiz/article-5874213/Did-Miley-Cyrus-Liam-Hemsworth-secretly-married.html	Did Miley Cyrus and Liam Hemsworth secretly get married?	284329075902926848	284332744559968256	284335412590297089	\[... more tweet IDs ...]

```



\*\*From politifact\_real.csv:\*\*

```

politifact14984	http://www.nfib-sbet.org/	National Federation of Independent Business	967132259869487105	967164368768196609	967215618687512581	\[... more tweet IDs ...]

```
 -->

# Dataset: FakeNewsNet (Unified Schema)

## Source
- **URL**: https://github.com/KaiDMML/FakeNewsNet
- **Clone Date**: [Your clone date]
- **SHA256**: `622c7162d8af12302528d7d80357763c8d1eb984f85c59ff1a61a0319d465577`
- **Archive Location**: `data/raw/fakenewsnet/original/FakeNewsNet_repo_snapshot.zip`

## Citation
Shu, K., Mahudeswaran, D., Wang, S., Lee, D., & Liu, H. (2020). "FakeNewsNet: A Data Repository with News Content, Social Context, and Spatiotemporal Information for Studying Fake News on Social Media." *Big Data*, 8(3), 171-188. [GitHub Repository](https://github.com/KaiDMML/FakeNewsNet)

## Dataset Overview

**Total articles: ~23,196**
- PolitiFact: ~1,000 articles (political news)
- GossipCop: ~22,000 articles (entertainment news)

**Key Features:**
- Fact-checked labels from professional fact-checkers
- Full article text available in many cases
- Rich temporal metadata (publication dates)
- Social context data (Twitter engagement)
- Dual domains (politics + entertainment)

## Unified Schema Fields

### Core Fields
| Field | Type | Description |
|-------|------|-------------|
| `id` | str | Unique article identifier |
| `dataset` | str | "fakenewsnet" |
| `split` | str | "train" / "val" / "test" |
| `claim_text` | str | Article headline/title |
| `article_text` | str | Full article body text |
| `label` | str | "true" / "false" |
| `label_confidence` | str | "gold" (fact-checked) |

### Content Metadata
| Field | Type | Description |
|-------|------|-------------|
| `content_status` | str | "full_article" / "partial" / "title_only" |
| `content_char_len` | int | Character count of article text |
| `source_dataset` | str | "politifact" / "gossipcop" |
| `is_hydrated` | bool | Whether article text was fetched |
| `lang` | str | Detected language code |

### URL & Source
| Field | Type | Description |
|-------|------|-------------|
| `news_url` | str | Original article URL |
| `source_domain` | str | Domain extracted from URL |
| `archive_url` | str | Wayback Machine URL (if used) |
| `is_archived` | bool | Whether content from archive |

### Fetch Metadata
| Field | Type | Description |
|-------|------|-------------|
| `fetch_status` | str | "success" / "existing_text" / "http_404" / etc. |
| `fetch_attempts` | int | Number of fetch attempts |
| `last_fetch_at` | str | ISO timestamp of last fetch |
| `ingested_at` | str | ISO timestamp of ingestion |
| `claim_norm_hash` | str | SHA1 hash of normalized title |

### Original Metadata (Preserved)
| Field | Type | Description |
|-------|------|-------------|
| `author` | str | Article author name |
| `publish_date` | str | Publication date/time |
| `source` | str | News source name |
| `top_img` | str | Featured image URL |
| `images` | list | All image URLs in article |
| `meta_description` | str | Article meta description |
| `meta_keywords` | str | Article meta keywords |
| `tags` | list | Article tags |

## Label Information

### Label Source
- **PolitiFact**: Fact-checking ratings from PolitiFact.com
  - Original ratings collapsed to binary
  - "Pants on Fire", "False", "Mostly False" → `false`
  - "True", "Mostly True" → `true`
  
- **GossipCop**: Credibility ratings from GossipCop.com
  - Binary scale: 0-4 (fake) vs 5-10 (real)
  - 0-4 → `false`
  - 5-10 → `true`

### Label Distribution
- **Balanced design**: ~50% fake, ~50% real
- **Gold standard**: Professional fact-checker verification
- `label_confidence`: "gold" for all articles

## File Structure

### Raw Data
```
data/raw/fakenewsnet/
├── politifact/
│   ├── fake/
│   │   └── *.json (~500 files)
│   └── real/
│       └── *.json (~500 files)
├── gossipcop/
│   ├── fake/
│   │   └── *.json (~11,000 files)
│   └── real/
│       └── *.json (~11,000 files)
└── original/
    └── FakeNewsNet_repo_snapshot.zip
```

### Processed Data
```
data/processed/fakenewsnet/
├── unified_fakenewsnet.parquet         # Full dataset
├── unified_fakenewsnet_sample.csv      # Sample (100 rows)
└── batches/
    └── hydrated_batch_*.parquet        # Intermediate batches
```

## Usage Example

```python
import pandas as pd

# Load unified FakeNewsNet data
df = pd.read_parquet('data/processed/fakenewsnet/unified_fakenewsnet.parquet')

# Basic statistics
print(f"Total articles: {len(df)}")
print(f"\nLabel distribution:")
print(df['label'].value_counts(normalize=True))

# Filter by domain
politifact = df[df['source_dataset'] == 'politifact']
gossipcop = df[df['source_dataset'] == 'gossipcop']

print(f"\nPolitiFact articles: {len(politifact)}")
print(f"GossipCop articles: {len(gossipcop)}")

# Filter by label
fake_news = df[df['label'] == 'false']
real_news = df[df['label'] == 'true']

# Filter by content status
full_articles = df[df['content_status'] == 'full_article']
print(f"\nFull articles: {len(full_articles)} ({len(full_articles)/len(df):.1%})")

# Temporal analysis (if publish_date available)
df['publish_date'] = pd.to_datetime(df['publish_date'], errors='coerce')
df_temporal = df.dropna(subset=['publish_date'])
print(f"\nArticles with dates: {len(df_temporal)}")
print(f"Date range: {df_temporal['publish_date'].min()} to {df_temporal['publish_date'].max()}")
```

## Processing Notes

### Text Availability
- **Many articles have full text** in original data
- Selective hydration: only fetch when text missing or <200 chars
- Some URLs are 4-5 years old (may be dead)
- Wayback Machine fallback for dead URLs

### Content Strategy
1. **Use existing text first** (priority)
2. **Hydrate URL** if text missing/short
3. **Try Wayback** if live URL fails
4. **Keep title only** as last resort

### Domain Characteristics
- **PolitiFact**: Political claims, policy statements
- **GossipCop**: Entertainment news, celebrity rumors
- **Imbalance**: GossipCop (22k) >> PolitiFact (1k)

### Temporal Considerations
- Publication dates span multiple years
- Date format varies (Unix epoch, ISO strings)
- Standardize to ISO 8601 for consistency

## Data Quality

### Challenges
- Text availability varies by article
- URL decay (4-5 years old)
- Domain imbalance (GossipCop >> PolitiFact)
- Date format inconsistency
- Some missing metadata (authors, tags)

### Recommendations
1. **Domain-specific models**: Train separate models for PolitiFact vs GossipCop
2. **Temporal analysis**: Use publication dates for time-series studies
3. **Cross-domain transfer**: Test generalization across politics/entertainment
4. **Social features**: Integrate Twitter engagement data if available
5. **Hydration monitoring**: Track fetch success rates by domain

## Reproduction Steps

### Clone and Archive
```bash
git clone https://github.com/KaiDMML/FakeNewsNet.git temp_fakenewsnet
mkdir -p data/raw/fakenewsnet/original
zip -r data/raw/fakenewsnet/original/FakeNewsNet_repo_snapshot.zip temp_fakenewsnet
sha256sum data/raw/fakenewsnet/original/FakeNewsNet_repo_snapshot.zip > data/raw/fakenewsnet/original/FakeNewsNet_SHA256.txt
rm -rf temp_fakenewsnet
unzip -d data/raw/fakenewsnet data/raw/fakenewsnet/original/FakeNewsNet_repo_snapshot.zip
```

### Process to Unified Schema
```bash
cd data/unified_schema/fakenewsnet

# Test with 100 articles
python fakenewsnet_unified.py --test

# Test with 1000 articles
python fakenewsnet_unified.py --max-articles 1000

# Process all articles
python fakenewsnet_unified.py
```

## Example Output Row

**PolitiFact Example:**
```python
{
    'id': 'politifact1234',
    'dataset': 'fakenewsnet',
    'split': 'train',
    'source_dataset': 'politifact',
    'claim_text': 'Politician Makes Outrageous Claim About Policy',
    'article_text': 'In a recent speech, the politician claimed that...',
    'label': 'false',
    'label_confidence': 'gold',
    'news_url': 'https://example.com/fake-news-article',
    'content_status': 'full_article',
    'content_char_len': 2453,
    'is_hydrated': True,
    'author': 'John Smith',
    'publish_date': '2018-05-15',
    'source': 'example.com',
    ...
}
```

**GossipCop Example:**
```python
{
    'id': 'gossipcop5678',
    'dataset': 'fakenewsnet',
    'split': 'train',
    'source_dataset': 'gossipcop',
    'claim_text': 'Actor Signs Major Deal with Studio',
    'article_text': 'Hollywood actor Jane Doe has signed a three-picture deal...',
    'label': 'true',
    'label_confidence': 'gold',
    'news_url': 'https://variety.com/2019/film/news/actor-signs-deal-1234567890/',
    'content_status': 'full_article',
    'content_char_len': 1876,
    'is_hydrated': True,
    'author': 'Entertainment Reporter',
    'publish_date': '2019-08-22T14:30:00Z',
    'source': 'variety.com',
    ...
}
```

## Performance Considerations

### Processing Time
- **23k articles** with hydration: ~2-4 hours (depending on network)
- **Batch processing** enabled: saves progress incrementally
- **Rate limiting**: 1 second per domain to avoid blocking

### Storage Requirements
- **Raw data**: ~500 MB
- **Processed parquet**: ~100-200 MB (compressed)
- **Batch files**: ~10-20 MB each

### Optimization Tips
1. Use `--max-articles` for testing
2. Enable batch saving for resumability
3. Adjust `max_workers` based on CPU cores
4. Monitor network for rate limiting

## License/Terms
Research use only. Check GitHub repository and original fact-checking sites for specific terms. Respect robots.txt when hydrating URLs.

## Related Datasets
- **LIAR**: Political statements with fine-grained labels
- **FakeHealth**: Health-specific misinformation with expert reviews
- **COAID**: COVID-19 misinformation claims and articles