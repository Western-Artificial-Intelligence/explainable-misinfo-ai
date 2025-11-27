<!-- \# Dataset: HoVer



\## Source

\- \*\*URL\*\*: https://github.com/hover-nlp/hover

\- \*\*Clone Date\*\*: 2025-10-19

\- \*\*SHA256\*\*: `\[check your HoVer\_SHA256.txt file]`

\- \*\*Archive Location\*\*: `data/raw/hover/original/HoVer\_repo\_snapshot.zip`



\## Citation

Jiang, Y., Bordes, A., Cancedda, N., Hoffman, M., \& Petrowski, B. (2020). "HoVer: A Dataset for Many-Hop Fact Extraction and Claim Verification." In \*Findings of the Association for Computational Linguistics: EMNLP 2020\* (pp. 3418–3430). \[arXiv:2011.03564](https://arxiv.org/abs/2011.03564)



\## Files \& Structure



\*\*data/raw/hover/data/hover/ contains 3 JSON files:\*\*

\- `hover\_train\_release\_v1.1.json` — 18,171 training claims with supporting evidence

\- `hover\_dev\_release\_v1.1.json` — 4,000 development claims with supporting evidence

\- `hover\_test\_release\_v1.1.json` — 4,000 test claims (labels held-out)

\- \*\*Total claims: 26,171\*\*



\## Schema (Key Files)



\### hover\_train\_release\_v1.1.json and hover\_dev\_release\_v1.1.json



JSON format where each entry contains:



| Field | Type | Notes |

|-------|------|-------|

| uid | str | Unique identifier (UUID) |

| claim | str | Factual claim to verify |

| supporting\_facts | list | List of \[Wikipedia article title, paragraph index] pairs |

| label | str | Verification label (SUPPORTED or NOT\_SUPPORTED) |

| num\_hops | int | Number of reasoning hops required (1, 2, or 3) |

| hpqa\_id | str | Reference ID to QAMR corpus |



\### hover\_test\_release\_v1.1.json



JSON format with reduced fields (labels held-out):



| Field | Type | Notes |

|-------|------|-------|

| uid | str | Unique identifier (UUID) |

| claim | str | Factual claim to verify |



\*\*Row counts:\*\*

\- Training: 18,171 entries

\- Development: 4,000 entries

\- Test: 4,000 entries (unlabeled)

\- \*\*Total: 26,171 entries\*\*



\## Labels (as provided)



2-class label distribution (train + dev):



\- `SUPPORTED` — Claim is verifiable from supporting Wikipedia facts

\- `NOT\_SUPPORTED` — Claim contradicts or cannot be verified from Wikipedia evidence



\## Notes/Quirks



\- \*\*Evidence-based verification:\*\* Claims must be verifiable using Wikipedia as knowledge source

\- \*\*Multi-hop reasoning:\*\* Requires linking multiple Wikipedia paragraphs (1-3 hops)

\- \*\*Test set unlabeled:\*\* Test set contains only claims without labels for evaluation

\- \*\*Wikipedia-dependent:\*\* All supporting facts reference specific Wikipedia articles and paragraphs

\- \*\*Structured evidence:\*\* Supporting facts provide exact locations for verification (article title + paragraph index)

\- \*\*Varying claim difficulty:\*\* num\_hops field indicates reasoning complexity (1, 2, or 3 hops)



\## License/Terms



HoVer dataset is provided for research use. Check GitHub repository for specific terms and attribution requirements.



\## Reproduction Steps



\### PowerShell (Windows)

```powershell

git clone https://github.com/hover-nlp/hover.git temp\_hover

mkdir data\\raw\\hover\\original -Force

Compress-Archive -Path temp\_hover\\\* -DestinationPath data\\raw\\hover\\original\\HoVer\_repo\_snapshot.zip

certutil -hashfile "data\\raw\\hover\\original\\HoVer\_repo\_snapshot.zip" SHA256 > data\\raw\\hover\\original\\HoVer\_SHA256.txt

Remove-Item -Recurse -Force temp\_hover

Expand-Archive -Path data\\raw\\hover\\original\\HoVer\_repo\_snapshot.zip -DestinationPath data\\raw\\hover

```



\### Bash/Linux/macOS

```bash

git clone https://github.com/hover-nlp/hover.git temp\_hover

mkdir -p data/raw/hover/original

zip -r data/raw/hover/original/HoVer\_repo\_snapshot.zip temp\_hover

sha256sum data/raw/hover/original/HoVer\_repo\_snapshot.zip > data/raw/hover/original/HoVer\_SHA256.txt

rm -rf temp\_hover

unzip -d data/raw/hover data/raw/hover/original/HoVer\_repo\_snapshot.zip

```



\## Example Rows



\*\*From hover\_train\_release\_v1.1.json:\*\*

```json

{

&nbsp; "uid": "330ca632-e83f-4011-b11b-0d0158145036",

&nbsp; "claim": "Skagen Painter Peder Severin Krøyer favored naturalism along with Theodor Esbern Philipsen and the artist Ossian Elgström studied with in the early 1900s.",

&nbsp; "supporting\_facts": \[\["Kristian Zahrtmann", 0], \["Kristian Zahrtmann", 1], \["Peder Severin Krøyer", 0]],

&nbsp; "label": "SUPPORTED",

&nbsp; "num\_hops": 2,

&nbsp; "hpqa\_id": "5abed82a5542993fe9a41d51"

}

```



\*\*From hover\_dev\_release\_v1.1.json:\*\*

```json

{

&nbsp; "uid": "042339bf-0374-4ab3-ab49-6df5f12d868e",

&nbsp; "claim": "The song recorded by Fergie that was produced by Polow da Don and was followed by Life Goes On was M.I.L.F.$.",

&nbsp; "supporting\_facts": \[\["Life Goes On (Fergie song)", 2], \["M.I.L.F. $", 1]],

&nbsp; "label": "SUPPORTED",

&nbsp; "num\_hops": 2,

&nbsp; "hpqa\_id": "5abed82a5542993fe9a41d51"

}

```



\*\*From hover\_test\_release\_v1.1.json (unlabeled):\*\*

```json

{

&nbsp; "uid": "062b6bde-c5e0-4dee-92f9-e2c01c7c32eb",

&nbsp; "claim": "The headquarters of the league the Basketball team Denver Pioneers play in are located in Elmhurst, Illinois."

}

```
 -->


# Dataset: HoVer (Unified Schema)

## Source
- **URL**: https://github.com/hover-nlp/hover
- **Clone Date**: [Your clone date]
- **SHA256**: [To be calculated]
- **Archive Location**: `data/raw/hover/original/HoVer_repo_snapshot.zip`

## Citation
Jiang, Y., Bordes, A., Cancedda, N., Hoffman, M., & Petrowski, B. (2020). "HoVer: A Dataset for Many-Hop Fact Extraction and Claim Verification." In *Findings of the Association for Computational Linguistics: EMNLP 2020* (pp. 3418–3430). [arXiv:2011.03564](https://arxiv.org/abs/2011.03564)

## Dataset Overview

**Total claims: 26,171**
- Train: 18,171 claims
- Dev: 4,000 claims
- Test: 4,000 claims (labels held-out)

**Key Features:**
- Multi-hop reasoning (1-3 hops)
- Wikipedia-grounded evidence
- Structured supporting facts
- Explicit reasoning chains
- Fact extraction + verification

## Unified Schema Fields

### Core Fields
| Field | Type | Description |
|-------|------|-------------|
| `id` | str | Unique claim identifier (UUID) |
| `dataset` | str | "hover" |
| `split` | str | "train" / "dev" / "test" |
| `claim_text` | str | Factual claim to verify |
| `label` | str | "true" / "false" (SUPPORTED / NOT_SUPPORTED) |
| `label_confidence` | str | "gold" (expert-annotated) |

### HoVer-Specific Fields
| Field | Type | Description |
|-------|------|-------------|
| `supporting_facts` | list | List of [Wikipedia title, paragraph idx] pairs |
| `num_hops` | int | Number of reasoning hops required (1, 2, or 3) |
| `hpqa_id` | str | Reference ID to QAMR corpus |

### Metadata Fields
| Field | Type | Description |
|-------|------|-------------|
| `content_status` | str | "title_only" (claims are short text) |
| `content_char_len` | int | Character count of claim |
| `claim_norm_hash` | str | SHA1 hash of normalized claim |
| `ingested_at` | str | ISO timestamp of ingestion |

## Label Information

### Label Mapping
- **SUPPORTED** → `true` (claim is verifiable from Wikipedia)
- **NOT_SUPPORTED** → `false` (claim contradicts or unverifiable)

### Label Distribution (Train + Dev)
Expected balanced distribution:
- **SUPPORTED** (~50%): Verifiable claims
- **NOT_SUPPORTED** (~50%): Contradicted/unverifiable claims

### Test Set
- Labels **held-out** for evaluation
- 4,000 test claims with `label = None`

## Multi-Hop Reasoning

### Hop Complexity
| Hops | Description | Example |
|------|-------------|---------|
| 1 | Single-step verification | "Paris is the capital of France" |
| 2 | Two-step reasoning | "The director of Titanic also directed Avatar" |
| 3 | Three-step reasoning | "The actor in Movie A starred with the director of Movie B's spouse" |

### Supporting Facts Structure
```python
supporting_facts = [
    ["Paris", 0],           # Wikipedia article "Paris", paragraph 0
    ["France", 2]           # Wikipedia article "France", paragraph 2
]
```

Each supporting fact points to:
- **Wikipedia article title**
- **Paragraph index** (0-indexed)

## File Structure

### Raw Data
```
data/raw/hover/
├── data/
│   └── hover/
│       ├── hover_train_release_v1.1.json    # 18,171 claims
│       ├── hover_dev_release_v1.1.json      # 4,000 claims
│       └── hover_test_release_v1.1.json     # 4,000 claims (unlabeled)
└── original/
    └── HoVer_repo_snapshot.zip
```

### Processed Data
```
data/processed/hover/
├── unified_hover_train.parquet              # Training set
├── unified_hover_dev.parquet                # Development set
├── unified_hover_test.parquet               # Test set (no labels)
├── unified_hover_sample.csv                 # Sample (100 rows)
└── label_statistics.txt                     # Label distribution stats
```

## Schema Differences from Other Datasets

### Key Differences
1. **No article text**: Claims are short factual statements
2. **Wikipedia grounding**: Evidence from Wikipedia, not news articles
3. **Structured evidence**: Explicit supporting fact locations
4. **No URL hydration needed**: Claims are self-contained
5. **Multi-hop reasoning**: Requires chaining multiple facts

### Content Status
- All HoVer entries have `content_status = "title_only"`
- No `article_text` field (claims are the content)
- No URL fetching required

## Usage Example

```python
import pandas as pd

# Load unified HoVer data
train_df = pd.read_parquet('data/processed/hover/unified_hover_train.parquet')
dev_df = pd.read_parquet('data/processed/hover/unified_hover_dev.parquet')
test_df = pd.read_parquet('data/processed/hover/unified_hover_test.parquet')

# Basic statistics
print(f"Train claims: {len(train_df)}")
print(f"Dev claims: {len(dev_df)}")
print(f"Test claims: {len(test_df)}")

# Label distribution (train + dev only)
print("\nLabel distribution:")
print(train_df['label'].value_counts(normalize=True))

# Filter by reasoning complexity
single_hop = train_df[train_df['num_hops'] == 1]
two_hop = train_df[train_df['num_hops'] == 2]
three_hop = train_df[train_df['num_hops'] == 3]

print(f"\n1-hop claims: {len(single_hop)} ({len(single_hop)/len(train_df):.1%})")
print(f"2-hop claims: {len(two_hop)} ({len(two_hop)/len(train_df):.1%})")
print(f"3-hop claims: {len(three_hop)} ({len(three_hop)/len(train_df):.1%})")

# Access supporting facts
example = train_df.iloc[0]
print(f"\nClaim: {example['claim_text']}")
print(f"Supporting facts: {example['supporting_facts']}")
print(f"Num hops: {example['num_hops']}")
print(f"Label: {example['label']}")
```

## Processing Notes

### No Hydration Required
- Claims are self-contained text
- No URLs to fetch
- No article parsing needed
- Evidence is in Wikipedia (external to dataset)

### Evidence Retrieval
To verify claims, you need:
1. **Wikipedia dump** (not included in HoVer)
2. **Evidence retrieval system** to fetch supporting paragraphs
3. **Multi-hop reasoning model** to chain facts

### Reasoning Complexity
- **1-hop**: Direct fact lookup
- **2-hop**: Connect two facts (entity bridging)
- **3-hop**: Chain three facts (complex reasoning)

## Data Quality

### Challenges
- No article text (different from news datasets)
- Requires Wikipedia for evidence retrieval
- Complex multi-hop reasoning needed
- Test labels held-out (cannot evaluate locally)
- Supporting facts point to Wikipedia versions (may change)

### Recommendations
1. **Evidence retrieval**: Build Wikipedia paragraph retrieval system
2. **Multi-hop models**: Use graph neural networks or iterative reasoning
3. **Complexity analysis**: Evaluate by num_hops separately
4. **Cross-dataset**: Combine with news datasets for robust models
5. **Explainability**: Use supporting_facts for interpretability

## Reproduction Steps

### Clone and Archive
```bash
git clone https://github.com/hover-nlp/hover.git temp_hover
mkdir -p data/raw/hover/original
zip -r data/raw/hover/original/HoVer_repo_snapshot.zip temp_hover
sha256sum data/raw/hover/original/HoVer_repo_snapshot.zip > data/raw/hover/original/HoVer_SHA256.txt
rm -rf temp_hover
unzip -d data/raw/hover data/raw/hover/original/HoVer_repo_snapshot.zip
```

### Process to Unified Schema
```bash
cd data/unified_schema/hover

# Process train split
python hover_unified.py --split train

# Process dev split
python hover_unified.py --split dev

# Process test split (no labels)
python hover_unified.py --split test

# Or process all splits
python hover_unified.py --all
```

## Example Output Row

**Training Example:**
```python
{
    'id': '330ca632-e83f-4011-b11b-0d0158145036',
    'dataset': 'hover',
    'split': 'train',
    'claim_text': 'Skagen Painter Peder Severin Krøyer favored naturalism along with Theodor Esbern Philipsen and the artist Ossian Elgström studied with in the early 1900s.',
    'label': 'true',  # SUPPORTED
    'label_confidence': 'gold',
    'supporting_facts': [
        ['Kristian Zahrtmann', 0],
        ['Kristian Zahrtmann', 1],
        ['Peder Severin Krøyer', 0]
    ],
    'num_hops': 2,
    'hpqa_id': '5abed82a5542993fe9a41d51',
    'content_status': 'title_only',
    'content_char_len': 156,
    'claim_norm_hash': 'a1b2c3d4e5f6...',
    'ingested_at': '2025-01-15T10:30:00Z'
}
```

**Test Example (No Label):**
```python
{
    'id': '062b6bde-c5e0-4dee-92f9-e2c01c7c32eb',
    'dataset': 'hover',
    'split': 'test',
    'claim_text': 'The headquarters of the league the Basketball team Denver Pioneers play in are located in Elmhurst, Illinois.',
    'label': None,  # Held-out for evaluation
    'label_confidence': None,
    'supporting_facts': None,
    'num_hops': None,
    'content_status': 'title_only',
    'content_char_len': 112,
    ...
}
```

## Integration with Wikipedia

### Wikipedia Requirements
HoVer requires Wikipedia for evidence retrieval:

1. **Wikipedia dump**: Download Wikipedia snapshot
2. **Paragraph extraction**: Index Wikipedia by article + paragraph
3. **Evidence retrieval**: Fetch paragraphs from supporting_facts

### Example Evidence Retrieval
```python
def get_supporting_evidence(claim_row, wiki_index):
    """Retrieve Wikipedia paragraphs for supporting facts."""
    evidence = []
    for article_title, para_idx in claim_row['supporting_facts']:
        paragraph = wiki_index.get(article_title, para_idx)
        evidence.append({
            'article': article_title,
            'paragraph_idx': para_idx,
            'text': paragraph
        })
    return evidence
```

## Model Architectures for HoVer

### Recommended Approaches
1. **Retrieval + Reasoning**: Two-stage pipeline
   - Stage 1: Retrieve relevant Wikipedia paragraphs
   - Stage 2: Multi-hop reasoning over retrieved facts
   
2. **Graph Neural Networks**: Model entities and relations
   - Nodes: Entities in claim and Wikipedia
   - Edges: Relations between entities
   - Reasoning: Message passing over graph

3. **Transformer Models**: End-to-end reasoning
   - Input: Claim + Retrieved paragraphs
   - Output: SUPPORTED / NOT_SUPPORTED
   - Attention: Learn to focus on relevant facts

4. **Iterative Reasoning**: Chain of thought
   - Hop 1: Find first supporting fact
   - Hop 2: Use Hop 1 to find second fact
   - Hop 3: Use Hop 2 to find third fact

## Performance Considerations

### Processing Time
- **26k claims** processing: <5 minutes (no hydration)
- Fast: No URL fetching or HTML parsing
- Instant: Claims are already clean text

### Storage Requirements
- **Raw data**: ~50 MB
- **Processed parquet**: ~10-20 MB
- Very lightweight compared to news datasets

## License/Terms
Research use only. Check GitHub repository for specific terms. Wikipedia content under CC BY-SA license.

## Related Datasets
- **FEVER**: Fact extraction and verification with Wikipedia
- **MultiHop-QA**: Multi-hop question answering
- **HotpotQA**: Diverse multi-hop questions with Wikipedia
- **FakeNewsNet**: News article verification
- **FakeHealth**: Health claim verification with expert reviews