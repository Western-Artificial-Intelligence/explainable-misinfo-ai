\# Dataset: CoAID



\## Source

\- \*\*URL\*\*: https://github.com/cuilimeng/CoAID

\- \*\*Clone Date\*\*: 2025-10-18

\- \*\*SHA256\*\*: `a056a22119b1dd294540f5d5c0faca2462705f87d0c68f35470003d2f7f2605d`

\- \*\*Archive Location\*\*: `data/raw/coaid/original/CoAID\_repo\_snapshot.zip`



\## Citation

Cui, L., \& Lee, S. (2020). "CoAID: COVID-19 Healthcare Misinformation Dataset." In \*Findings of the Association for Computational Linguistics: EMNLP 2020\* (pp. 1496–1508). \[ACL Anthology](https://aclanthology.org/2020.findings-emnlp.138/)



\## Files Pulled

The CoAID repository contains COVID-19 misinformation data organized by time snapshots (4 total):

\- `05-01-2020/` — May 1, 2020 snapshot (12 CSV files)

\- `07-01-2020/` — July 1, 2020 snapshot (12 CSV files)

\- `09-01-2020/` — September 1, 2020 snapshot (9 CSV files)

\- `11-01-2020/` — November 1, 2020 snapshot (9 CSV files)

\- `README.md` — Repository documentation



\*\*File naming pattern:\*\* `\[Claim|News]\[Real|Fake]COVID-19\[\_tweets]\[\_replies].csv`



Total files: 42 CSV files across all snapshots

Total rows: ~237,000+ entries (mix of claims, news articles, tweets, and replies)



\## Schema (Key Files)



\### Claim/News CSV files (e.g., ClaimFakeCOVID-19.csv, NewsRealCOVID-19.csv)

| Column | Type | Notes |

|--------|------|-------|

| claim\_id / news\_id | int | Unique identifier |

| text | str | Claim or article text |

| created\_at | str | Creation timestamp |

| label | str | Ground-truth label (real or fake) |

| source | str | Source URL |



\### Tweet/Reply CSV files (e.g., ClaimFakeCOVID-19\_tweets.csv)

| Column | Type | Notes |

|--------|------|-------|

| tweet\_id | int | Tweet identifier |

| text | str | Tweet content |

| created\_at | str | Tweet timestamp |

| user\_id | int | Twitter user ID |

| retweet\_count | int | Retweet count |

| favorite\_count | int | Like/favorite count |

| reply\_count | int | Reply count |

| label | str | Tweet label (real or fake) |



\### Sample row counts (05-01-2020 snapshot):

\- ClaimFakeCOVID-19.csv: 27 rows

\- ClaimFakeCOVID-19\_tweets.csv: 457 rows

\- ClaimFakeCOVID-19\_tweets\_replies.csv: 623 rows

\- ClaimRealCOVID-19.csv: 166 rows

\- ClaimRealCOVID-19\_tweets.csv: 6,342 rows

\- ClaimRealCOVID-19\_tweets\_replies.csv: 9,764 rows

\- NewsFakeCOVID-19.csv: 572 rows

\- NewsFakeCOVID-19\_tweets.csv: 9,218 rows

\- NewsFakeCOVID-19\_tweets\_replies.csv: 5,721 rows

\- NewsRealCOVID-19.csv: 1,590 rows

\- NewsRealCOVID-19\_tweets.csv: 87,324 rows

\- NewsRealCOVID-19\_tweets\_replies.csv: 64,115 rows



\## Labels (as provided)

Binary label per file structure:

\- \*\*Claim files:\*\* Organized by Real vs. Fake (implicit in filename)

\- \*\*News files:\*\* Organized by Real vs. Fake (implicit in filename)

\- \*\*Tweets:\*\* Text label field (pattern varies by snapshot)



\## Notes/Quirks

\- \*\*Binary implicit labels:\*\* Label determined by filename (Fake vs. Real), not always explicit in CSV

\- \*\*Multiple modalities:\*\* Claims, articles, tweets, and replies all mixed; track source type

\- \*\*Time-series data:\*\* 4 snapshots across May-November 2020; COVID data evolves, claims get debunked

\- \*\*Engagement metrics:\*\* Retweets, favorites, replies provide social signal; not identical to veracity

\- \*\*Missing columns across snapshots:\*\* 09-01-2020 and 11-01-2020 have fewer files (data may consolidate)

\- \*\*Social media platform data:\*\* Primarily Twitter; timestamps in UTC

\- \*\*Character encoding:\*\* Some non-ASCII characters (emojis, Unicode) present in tweets



\## License/Terms

CoAID dataset is provided for research use. Check the GitHub repository README for specific terms and citation requirements.



\## Reproduction Steps



\### PowerShell (Windows)

```powershell

git clone https://github.com/cuilimeng/CoAID.git temp\_coaid

mkdir data\\raw\\coaid\\original -Force

Compress-Archive -Path temp\_coaid\\\* -DestinationPath data\\raw\\coaid\\original\\CoAID\_repo\_snapshot.zip

certutil -hashfile "data\\raw\\coaid\\original\\CoAID\_repo\_snapshot.zip" SHA256 > data\\raw\\coaid\\original\\CoAID\_SHA256.txt

Remove-Item -Recurse -Force temp\_coaid

Expand-Archive data\\raw\\coaid\\original\\CoAID\_repo\_snapshot.zip -DestinationPath temp\_check

Copy-Item -Path temp\_check\\\* -Destination data\\raw\\coaid -Recurse

Remove-Item -Recurse -Force temp\_check

```



\### Bash/Linux/macOS

```bash

git clone https://github.com/cuilimeng/CoAID.git temp\_coaid

mkdir -p data/raw/coaid/original

zip -r data/raw/coaid/original/CoAID\_repo\_snapshot.zip temp\_coaid

sha256sum data/raw/coaid/original/CoAID\_repo\_snapshot.zip > data/raw/coaid/original/CoAID\_SHA256.txt

rm -rf temp\_coaid

unzip -d data/raw/coaid data/raw/coaid/original/CoAID\_repo\_snapshot.zip

```



\## Example Row (Sanitized)



\*\*From ClaimFakeCOVID-19.csv (05-01-2020):\*\*

