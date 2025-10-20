\# Dataset: FakeNewsNet



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

