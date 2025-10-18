\# Dataset: FakeHealth



\## Source

\- \*\*URL\*\*: https://github.com/EnyanDai/FakeHealth

\- \*\*Clone Date\*\*: 2025-10-18

\- \*\*SHA256\*\*: `9ebe53431811b0ba7c0bff7643be739699b64581b2e5e43e0a45e8725109b62c`

\- \*\*Archive Location\*\*: `data/raw/fakehealth/original/FakeHealth\_repo\_snapshot.zip`



\## Citation

Dai, E., Zhao, Y., Zhu, H., Tian, Z., Tan, Z., \& Huang, L. (2022). "Towards Fake News Detection: A Content, Social, and Political View." In \*Proceedings of the 15th ACM International Conference on Web Search and Data Mining\* (WSDM '22), pp. 144–154. Extended work: Dai, E., \& Wang, S. (2021). "Ginger Cannot Cure Cancer: Analyzing Health Claims in News." arXiv preprint arXiv:2104.05808. \[GitHub Repository](https://github.com/EnyanDai/FakeHealth)



\## Files Pulled

FakeHealth contains health-related news content organized by content type:

\- Extracted articles CSV: `data/raw/fakehealth/fakehealth\_articles.csv`

\- Repository structure: `dataset/content/\[HealthRelease|HealthStory]/`

\- Original JSON files: 2,237 individual JSON articles

\- Repository snapshot: `data/raw/fakehealth/original/FakeHealth\_repo\_snapshot.zip`



\*\*Content breakdown:\*\*

\- HealthRelease articles: 599 JSON files

\- HealthStory articles: 1,638 JSON files

\- \*\*Total articles: 2,237\*\*



\## Schema (Key Files)



\### fakehealth\_articles.csv (Extracted/Flattened)

| Column | Type | Notes |

|--------|------|-------|

| id | str | Article ID (JSON filename without extension) |

| content\_type | str | "HealthRelease" or "HealthStory" |

| title | str | Article headline/title |

| text | str | Full article body/content |

| url | str | Original article URL (often archived URL) |

| publish\_date | float | Unix timestamp of publication |

| source | str | Content source (e.g., "web.archive.org") |

| keywords | str | Comma-separated keywords |

| authors | str | Comma-separated author names |

| top\_img | str | Top/featured image URL |

| canonical\_link | str | Canonical URL reference |

| summary | str | Article summary (often empty) |

| meta\_description | str | Meta description from article page |

| meta\_keywords | str | Meta keywords from page |

| meta\_date | str | Metadata date string |

| meta\_journal | str | Associated journal/publication |

| meta\_type | str | Article type (e.g., "research", "news") |

| meta\_institution | str | Associated institution |

| meta\_region | str | Geographic region code |



\*\*Article counts:\*\*

\- Total: 2,237 articles

\- HealthRelease: 599 articles

\- HealthStory: 1,638 articles



\*\*Text statistics:\*\*

\- Min length: 15 characters

\- Max length: 29,318 characters

\- Mean length: 4,133 characters



\## Labels (as provided)

\*\*NOTE:\*\* Labels are NOT included in the extracted CSV. The original repository structure includes:

\- `dataset/reviews/` — Expert veracity annotations (was empty in current clone)

\- `dataset/engagements/` — Social media engagement metrics (was empty in current clone)



The current extracted dataset contains \*\*article content only\*\* without veracity labels. Labels would need to be obtained from:

1\. Original review JSON files (if complete repo clone)

2\. Manual annotation for preprocessing

3\. Cross-referencing with original sources



\## Notes/Quirks

\- \*\*No labels in extracted CSV:\*\* Veracity labels were not available in the current extraction

\- \*\*Missing engagement/review data:\*\* The reviews/ and engagements/ directories were empty in the clone

\- \*\*Archived URLs:\*\* Many articles sourced from web.archive.org, not original URLs

\- \*\*Long articles:\*\* Mean 4,133 chars; some exceed 25,000 chars; may need chunking

\- \*\*Medical jargon:\*\* Articles contain domain-specific health terminology, chemical names, disease names

\- \*\*Sparse metadata:\*\* Some rows missing keywords, authors, or metadata fields

\- \*\*HTML remnants possible:\*\* No observed in sample, but potential for encoding artifacts

\- \*\*Date format inconsistency:\*\* Timestamps in multiple formats (Unix, ISO 8601, custom strings)

\- \*\*Content type imbalance:\*\* HealthStory (1,638) is ~2.7x more common than HealthRelease (599)



\## License/Terms

FakeHealth dataset is provided for research use. Check the GitHub repository README for specific terms, attribution requirements, and ethical considerations for health-related misinformation data.



\## Reproduction Steps



\### PowerShell (Windows)

```powershell

git clone https://github.com/EnyanDai/FakeHealth.git temp\_fakehealth

mkdir data\\raw\\fakehealth\\original -Force

Compress-Archive -Path temp\_fakehealth\\\* -DestinationPath data\\raw\\fakehealth\\original\\FakeHealth\_repo\_snapshot.zip

certutil -hashfile "data\\raw\\fakehealth\\original\\FakeHealth\_repo\_snapshot.zip" SHA256 > data\\raw\\fakehealth\\original\\FakeHealth\_SHA256.txt

Remove-Item -Recurse -Force temp\_fakehealth

Expand-Archive -Path data\\raw\\fakehealth\\original\\FakeHealth\_repo\_snapshot.zip -DestinationPath data\\raw\\fakehealth

python scripts/prepare\_fakehealth.py

```



\### Bash/Linux/macOS

```bash

git clone https://github.com/EnyanDai/FakeHealth.git temp\_fakehealth

mkdir -p data/raw/fakehealth/original

zip -r data/raw/fakehealth/original/FakeHealth\_repo\_snapshot.zip temp\_fakehealth

sha256sum data/raw/fakehealth/original/FakeHealth\_repo\_snapshot.zip > data/raw/fakehealth/original/FakeHealth\_SHA256.txt

rm -rf temp\_fakehealth

unzip -d data/raw/fakehealth data/raw/fakehealth/original/FakeHealth\_repo\_snapshot.zip

python scripts/prepare\_fakehealth.py

```



\## Example Row (Sanitized)



\*\*From fakehealth\_articles.csv:\*\*

