# Dataset: FakeHealth

## Source
- **URL**: https://github.com/EnyanDai/FakeHealth
- **Clone Date**: 2025-10-18
- **SHA256**: `9ebe53431811b0ba7c0bff7643be739699b64581b2e5e43e0a45e8725109b62c`
- **Archive Location**: `data/raw/fakehealth/original/FakeHealth_repo_snapshot.zip`

## Citation
Dai, E., Zhao, Y., Zhu, H., Tian, Z., Tan, Z., & Huang, L. (2022). "Towards Fake News Detection: A Content, Social, and Political View." In *Proceedings of the 15th ACM International Conference on Web Search and Data Mining* (WSDM '22), pp. 144–154. Extended work: Dai, E., & Wang, S. (2021). "Ginger Cannot Cure Cancer: Analyzing Health Claims in News." arXiv preprint arXiv:2104.05808. [GitHub Repository](https://github.com/EnyanDai/FakeHealth)

## Files & Structure
FakeHealth repository organized into three main directories:

**Content (Individual article JSON files):**
- `dataset/content/HealthRelease/` — 606 health news release JSON files (`news_reviews_00000.json` to `news_reviews_00605.json`)
- `dataset/content/HealthStory/` — 1,700 health story JSON files (`story_reviews_00000.json` to `story_reviews_01699.json`)
- **Total articles: 2,306**

**Reviews (Expert evaluation aggregates):**
- `dataset/reviews/HealthRelease.json` — Array of 606 expert review objects
- `dataset/reviews/HealthStory.json` — Array of expert review objects

**Engagements (Social media mappings):**
- `dataset/engagements/HealthRelease.json` — Tweet ID mappings for HealthRelease articles
- `dataset/engagements/HealthStory.json` — Tweet ID mappings for HealthStory articles

- **Original archive**: `data/raw/fakehealth/original/FakeHealth_repo_snapshot.zip`

## Schema

### Content JSON files (dataset/content/HealthRelease/*.json and dataset/content/HealthStory/*.json)

Each article stored as individual JSON with these fields:

| Field | Type | Notes |
|-------|------|-------|
| url | str | Original article URL (often from web.archive.org) |
| text | str | Full article body/content |
| title | str | Article headline |
| publish_date | float | Unix timestamp of publication date |
| source | str | Content source (e.g., "https://web.archive.org") |
| canonical_link | str | Canonical URL reference |
| keywords | list | Article keywords (often empty) |
| authors | list | Author names (often empty) |
| top_img | str | Top/featured image URL |
| summary | str | Article summary (usually empty) |
| images | list | Array of image URLs in article |
| movies | list | Array of videos (usually empty) |
| meta_data | dict | Nested metadata (viewport, description, keywords, date, journal, type, institution, region, twitter, og, fb) |

**File counts:**
- HealthRelease: 606 files
- HealthStory: 1,700 files
- **Total: 2,306 articles**

**Text statistics:**
- Min: 15 characters
- Max: 29,318 characters
- Mean: 4,133 characters

### Reviews JSON files (dataset/reviews/HealthRelease.json and dataset/reviews/HealthStory.json)

Arrays of expert review objects from HealthNewsReview.org:

| Field | Type | Notes |
|-------|------|-------|
| news_id | str | Article identifier (e.g., "news_reviews_00000") |
| link | str | HealthNewsReview review URL |
| title | str | Review title |
| description | str | Brief review summary |
| original_title | str | Original article headline |
| rating | int | Overall quality rating |
| reviewers | list | Expert reviewer names |
| category | str | Content category (e.g., "University news release") |
| tags | list | Topic tags |
| source_link | str | Original article source URL |
| summary | dict | Detailed review with "Our Review Summary" and "Why This Matters" |
| criteria | list | Array of 10 Q&A evaluation items (question, answer, explanation) |
| news_source | str | Organization/source of article |

**Review counts:**
- HealthRelease.json: 606 review objects
- HealthStory.json: Partial coverage (not all 1,700 stories have reviews)

**10 evaluation criteria:**
1. Are costs adequately discussed?
2. Are benefits quantified?
3. Are harms explained?
4. Does it grasp quality of evidence?
5. Does it commit disease-mongering?
6. Are funding sources & conflicts disclosed?
7. Does it compare with alternatives?
8. Is availability established?
9. Is true novelty established?
10. Does it use unjustifiable sensational language?

### Engagements JSON files (dataset/engagements/HealthRelease.json and dataset/engagements/HealthStory.json)

Mapping of article IDs to associated tweets:
```json
{
  "news_reviews_00000": {"tweets": [1075103588148334592, 1075507329322418181, ...]},
  "story_reviews_00001": {"tweets": [1074823300977713154, ...]}
}
```

**Important:** Contains ONLY tweet ID mappings, NOT tweet text or engagement metrics (likes, retweets, comments). Full tweet data requires Twitter API access (deprecated/restricted).

## Labels & Veracity

No explicit binary labels in dataset. Expert reviews from HealthNewsReview.org provide structured evaluation through:
- Individual criterion answers ("Satisfactory" / "Not Satisfactory")
- Overall rating scores  
- Detailed explanations and reasoning

Labels would need to be derived from:
1. Aggregating review criterion answers
2. Manual annotation
3. Cross-referencing with external fact-check sources

## Notes/Quirks
- **Incomplete review coverage:** 606 HealthRelease reviews exist; HealthStory reviews partial
- **Expert structured review:** 10-point Q&A criteria from HealthNewsReview.org, not binary labels
- **Engagement incomplete:** Tweet ID mappings only; actual tweet content/metrics missing
- **Long articles:** Mean 4,133 chars; some >25k chars; may need chunking
- **Medical terminology:** Domain-specific health jargon, drug names, disease references
- **Archived URLs:** Many articles from web.archive.org, not originals
- **Sparse metadata:** Some articles missing keywords, authors, metadata
- **Date format inconsistency:** Unix epoch and ISO strings mixed; standardize before use
- **Content imbalance:** HealthStory (1,700) ~2.8x more common than HealthRelease (606)

## License/Terms
Research use only. Health news reviews from HealthNewsReview.org used under their terms. Check GitHub README for specific terms and ethical considerations for health misinformation data.

## Reproduction Steps

### PowerShell (Windows)
```powershell
git clone https://github.com/EnyanDai/FakeHealth.git temp_fakehealth
mkdir data\raw\fakehealth\original -Force
Compress-Archive -Path temp_fakehealth\* -DestinationPath data\raw\fakehealth\original\FakeHealth_repo_snapshot.zip
certutil -hashfile "data\raw\fakehealth\original\FakeHealth_repo_snapshot.zip" SHA256 > data\raw\fakehealth\original\FakeHealth_SHA256.txt
Remove-Item -Recurse -Force temp_fakehealth
Expand-Archive -Path data\raw\fakehealth\original\FakeHealth_repo_snapshot.zip -DestinationPath data\raw\fakehealth
```

### Bash/Linux/macOS
```bash
git clone https://github.com/EnyanDai/FakeHealth.git temp_fakehealth
mkdir -p data/raw/fakehealth/original
zip -r data/raw/fakehealth/original/FakeHealth_repo_snapshot.zip temp_fakehealth
sha256sum data/raw/fakehealth/original/FakeHealth_repo_snapshot.zip > data/raw/fakehealth/original/FakeHealth_SHA256.txt
rm -rf temp_fakehealth
unzip -d data/raw/fakehealth data/raw/fakehealth/original/FakeHealth_repo_snapshot.zip
```

## Example Rows

**From dataset/content/HealthRelease/news_reviews_00000.json:**
```json
{
  "url": "https://web.archive.org/web/20181217203805/https://www.eurekalert.org/pub_releases/2018-12/b-src121718.php",
  "title": "Scientists report CRISPR restores effectiveness of lung cancer treatment",
  "text": "Wilmington, DE, December 17, 2018 - The CRISPR-Cas9 gene editing system may be able to restore effectiveness...",
  "publish_date": 1546059600.0,
  "meta_data": {"journal": "Molecular Therapy Oncolytics", "type": "research", "institution": "Christiana Care Health System"}
}
```

**From dataset/reviews/HealthRelease.json:**
```json
{
  "news_id": "news_reviews_00000",
  "title": "Scientists report CRISPR restores effectiveness of lung cancer treatment",
  "rating": 2,
  "reviewers": ["Andrew Holtz, MPH", "Yoni Freedhoff, MD"],
  "criteria": [{"question": "Does the news release adequately discuss costs?", "answer": "Not Satisfactory"}]
}
```

**From dataset/engagements/HealthRelease.json:**
```json
{
  "news_reviews_00000": {"tweets": [1075103588148334592, 1075507329322418181]}
}
```