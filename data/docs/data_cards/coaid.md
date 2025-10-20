# Dataset: CoAID

## Source
- **URL**: https://github.com/cuilimeng/CoAID
- **Clone Date**: 2025-10-18
- **SHA256**: `a056a22119b1dd294540f5d5c0faca2462705f87d0c68f35470003d2f7f2605d`
- **Archive Location**: `data/raw/coaid/original/CoAID_repo_snapshot.zip`

## Citation
Cui, L., & Lee, S. (2020). "CoAID: COVID-19 Healthcare Misinformation Dataset." In *Findings of the Association for Computational Linguistics: EMNLP 2020* (pp. 1496–1508). [ACL Anthology](https://aclanthology.org/2020.findings-emnlp.138/)

## Files Pulled
The CoAID repository contains COVID-19 misinformation data organized by time snapshots (4 total):
- `05-01-2020/` — May 1, 2020 snapshot (12 CSV files)
- `07-01-2020/` — July 1, 2020 snapshot (12 CSV files)
- `09-01-2020/` — September 1, 2020 snapshot (9 CSV files)
- `11-01-2020/` — November 1, 2020 snapshot (9 CSV files)
- `README.md` — Repository documentation

**File naming pattern:** `[Claim|News][Real|Fake]COVID-19[_tweets][_replies].csv`

Total files: 42 CSV files across all snapshots
Total rows: ~237,000+ entries (mix of claims, news articles, tweets, and replies)

## Schema (Key Files)

### Claim CSV files (e.g., ClaimFakeCOVID-19.csv, ClaimRealCOVID-19.csv)
| Column | Type | Notes |
|--------|------|-------|
| Unnamed: 0 | int | Index identifier |
| fact_check_url | str | Fact-check/verification source URL |
| news_url | str | Original claim/news source URL |
| title | str | Claim title or headline |

**Label encoded in filename:** "Fake" or "Real" (implicit)

### News CSV files (e.g., NewsFakeCOVID-19.csv, NewsRealCOVID-19.csv)
| Column | Type | Notes |
|--------|------|-------|
| Unnamed: 0 | int | Index identifier |
| type | str | Article type (varies by snapshot) |
| fact_check_url | str | Fact-check/verification source |
| archive | str | Archive URL (if available) |
| news_url / news_url2-5 | str | Original article URLs (up to 5 mirrors) |
| title | str | Article headline |
| newstitle | str | Alternative title |
| content | str | Full article text |
| abstract | str | Article summary |
| publish_date | str | Publication date |
| meta_keywords | str | Article keywords |

**Label encoded in filename:** "Fake" or "Real" (implicit)

### Tweet mapping CSV files (e.g., ClaimFakeCOVID-19_tweets.csv)
| Column | Type | Notes |
|--------|------|-------|
| index | int | Foreign key to claim/news ID |
| tweet_id | int | Twitter tweet identifier |

**Important:** Contains only tweet IDs and links to claims. Does NOT include tweet text, timestamps, or engagement metrics. Full tweet data requires Twitter API access (deprecated/restricted).

### Tweet reply mapping CSV files (e.g., ClaimFakeCOVID-19_tweets_replies.csv)
| Column | Type | Notes |
|--------|------|-------|
| news_id | int | Foreign key to news/claim ID |
| tweet_id | int | Parent tweet identifier |
| reply_id | int | Reply/child tweet identifier |

**Important:** Contains only tweet ID mappings. Does NOT include reply text or engagement data.

### Sample row counts (05-01-2020 snapshot):
- ClaimFakeCOVID-19.csv: 27 rows
- ClaimFakeCOVID-19_tweets.csv: 457 rows
- ClaimFakeCOVID-19_tweets_replies.csv: 623 rows
- ClaimRealCOVID-19.csv: 166 rows
- ClaimRealCOVID-19_tweets.csv: 6,342 rows
- ClaimRealCOVID-19_tweets_replies.csv: 9,764 rows
- NewsFakeCOVID-19.csv: 572 rows
- NewsFakeCOVID-19_tweets.csv: 9,218 rows
- NewsFakeCOVID-19_tweets_replies.csv: 5,721 rows
- NewsRealCOVID-19.csv: 1,590 rows
- NewsRealCOVID-19_tweets.csv: 87,324 rows
- NewsRealCOVID-19_tweets_replies.csv: 64,115 rows

## Labels (as provided)
Binary label per file structure:
- **Claim files:** Organized by Real vs. Fake (implicit in filename)
- **News files:** Organized by Real vs. Fake (implicit in filename)
- **Tweets:** Text label field (pattern varies by snapshot)

## Notes/Quirks
- **Binary implicit labels:** Label determined by filename (Fake vs. Real), not always explicit in CSV
- **Multiple modalities:** Claims, articles, tweets, and replies all mixed; track source type
- **Time-series data:** 4 snapshots across May-November 2020; COVID data evolves, claims get debunked
- **Engagement metrics:** Retweets, favorites, replies provide social signal; not identical to veracity
- **Missing columns across snapshots:** 09-01-2020 and 11-01-2020 have fewer files (data may consolidate)
- **Social media platform data:** Primarily Twitter; timestamps in UTC
- **Character encoding:** Some non-ASCII characters (emojis, Unicode) present in tweets

## License/Terms
CoAID dataset is provided for research use. Check the GitHub repository README for specific terms and citation requirements.

## Reproduction Steps

### PowerShell (Windows)
```powershell
git clone https://github.com/cuilimeng/CoAID.git temp_coaid
mkdir data\raw\coaid\original -Force
Compress-Archive -Path temp_coaid\* -DestinationPath data\raw\coaid\original\CoAID_repo_snapshot.zip
certutil -hashfile "data\raw\coaid\original\CoAID_repo_snapshot.zip" SHA256 > data\raw\coaid\original\CoAID_SHA256.txt
Remove-Item -Recurse -Force temp_coaid
Expand-Archive data\raw\coaid\original\CoAID_repo_snapshot.zip -DestinationPath temp_check
Copy-Item -Path temp_check\* -Destination data\raw\coaid -Recurse
Remove-Item -Recurse -Force temp_check
```

### Bash/Linux/macOS
```bash
git clone https://github.com/cuilimeng/CoAID.git temp_coaid
mkdir -p data/raw/coaid/original
zip -r data/raw/coaid/original/CoAID_repo_snapshot.zip temp_coaid
sha256sum data/raw/coaid/original/CoAID_repo_snapshot.zip > data/raw/coaid/original/CoAID_SHA256.txt
rm -rf temp_coaid
unzip -d data/raw/coaid data/raw/coaid/original/CoAID_repo_snapshot.zip
```

## Example Row (Sanitized)

**From ClaimFakeCOVID-19.csv (05-01-2020):**