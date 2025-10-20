\# Datasets Overview



This document provides a high-level index of all three misinformation datasets used in the Explainable Misinfo AI project.



\## Summary Table



| Dataset | Type | Total Size | Source Type | Location | Data Card |

|---------|------|-----------|-------------|----------|-----------|

| LIAR | Political fact-checks | 12,788 statements | Web archive (PolitiFact) | `data/raw/liar/` | \[liar.md](data\_cards/liar.md) |

| CoAID | COVID-19 healthcare misinfo | ~237k entries across 4 snapshots | GitHub repository (Twitter data) | `data/raw/coaid/` | \[coaid.md](data\_cards/coaid.md) |

| FakeHealth | Health news \& articles | 2,237 articles | GitHub repository (archived news) | `data/raw/fakehealth/` | \[fakehealth.md](data\_cards/fakehealth.md) |



\## Quick Reference



\### LIAR (Political Claims)

\- \*\*Size:\*\* 12,788 total statements (train: 10,239 / val: 1,283 / test: 1,266)

\- \*\*Format:\*\* TSV (tab-separated values)

\- \*\*Labels:\*\* 6-class (true, mostly-true, half-true, barely-true, false, pants-on-fire)

\- \*\*Key files:\*\* `train.tsv`, `valid.tsv`, `test.tsv`

\- \*\*Original source:\*\* https://www.cs.ucsb.edu/~william/data/liar\_dataset.zip

\- \*\*SHA256:\*\* `611c1addad919743dde15822b87a60bfb760d8f85597f25289e34621800654c7`

\- \*\*Citation:\*\* Wang (2017) ACL



\### CoAID (COVID-19 Healthcare Misinformation)

\- \*\*Size:\*\* ~237,000 entries total (mix of claims, articles, tweets, replies) across 4 time snapshots

\- \*\*Format:\*\* CSV (multiple files per snapshot)

\- \*\*Snapshots:\*\* 05-01-2020, 07-01-2020, 09-01-2020, 11-01-2020

\- \*\*File types per snapshot:\*\* Claims (Fake/Real), News (Fake/Real), Tweets, Tweet replies

\- \*\*Labels:\*\* Binary (Real/Fake) encoded in filename

\- \*\*Key files:\*\* `\[Claim|News]\[Real|Fake]COVID-19\[\_tweets]\[\_replies].csv`

\- \*\*Original source:\*\* https://github.com/cuilimeng/CoAID

\- \*\*SHA256:\*\* `a056a22119b1dd294540f5d5c0faca2462705f87d0c68f35470003d2f7f2605d`

\- \*\*Citation:\*\* Cui \& Lee (2020) ACL Findings



\### FakeHealth (Health News \& Articles)

\- \*\*Size:\*\* 2,237 health news articles (HealthRelease: 599 / HealthStory: 1,638)

\- \*\*Format:\*\* JSON (individual files, extracted to CSV)

\- \*\*Content types:\*\* HealthRelease (health/medical releases), HealthStory (news stories)

\- \*\*Text statistics:\*\* Mean length 4,133 chars (range: 15–29,318 chars)

\- \*\*Labels:\*\* NOT AVAILABLE in current extraction (reviews/engagements directories were empty)

\- \*\*Key files:\*\* `data/raw/fakehealth/fakehealth\_articles.csv`

\- \*\*Original source:\*\* https://github.com/EnyanDai/FakeHealth

\- \*\*SHA256:\*\* `9ebe53431811b0ba7c0bff7643be739699b64581b2e5e43e0a45e8725109b62c`

\- \*\*Citation:\*\* Dai et al. (2021, 2022)



\## Data Organization

