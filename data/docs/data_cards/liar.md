# Dataset: LIAR

## Source
- **URL**: https://www.cs.ucsb.edu/~william/data/liar_dataset.zip
- **Date Downloaded**: 2025-10-18
- **SHA256**: `611c1addad919743dde15822b87a60bfb760d8f85597f25289e34621800654c7`
- **Archive Location**: `data/raw/liar/original/liar.zip`

## Citation
Wang, W. Y. (2017). "Liar, Liar Pants on Fire: A New Benchmark Dataset for Fake News Detection." *Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (ACL)*, pp. 422–426. [ACL Anthology](https://aclanthology.org/P17-2067/)

## Files Pulled
- `train.tsv` — 10,269 rows; training set of labeled statements
- `valid.tsv` — 1,284 rows; validation set
- `test.tsv` — 1,283 rows; test set
- `README` — Original dataset documentation

## Schema (Key Files)

### train.tsv / valid.tsv / test.tsv

Tab-separated format. **Note:** Files do NOT include header row. Column order (left to right) is as follows:

| Position | Column | Type | Notes |
|----------|--------|------|-------|
| 1 | claim_id | str | Unique identifier (JSON filename, e.g., "2635.json") |
| 2 | label | str | Truthfulness label (6-class) |
| 3 | statement | str | The factual claim being checked |
| 4 | subject | str | Topic categories (comma-separated, e.g., "abortion") |
| 5 | speaker | str | Person making the statement |
| 6 | job_title | str | Speaker's role/position |
| 7 | state_info | str | Geographic context (state abbreviation) |
| 8 | party_affiliation | str | Political party (lowercase, e.g., "republican") |
| 9 | barely_true_counts | int | Count of "barely true" PolitiFact labels |
| 10 | false_counts | int | Count of "false" PolitiFact labels |
| 11 | half_true_counts | int | Count of "half true" PolitiFact labels |
| 12 | mostly_true_counts | int | Count of "mostly true" PolitiFact labels |
| 13 | pants_fire_counts | int | Count of "pants fire" PolitiFact labels |
| 14 | context | str | Reference/context (URL, article excerpt, or event description) |

**Row counts:**
- train.tsv: 10,269 rows
- valid.tsv: 1,284 rows
- test.tsv: 1,283 rows
- **Total: 12,836 rows**

## Labels (as provided)
6-class label distribution:
- `true` — Factually accurate statements
- `mostly-true` — Mostly accurate with minor inaccuracy
- `half-true` — Mixed accuracy, true and false elements
- `barely-true` — Mostly false with minor truth
- `false` — Factually inaccurate
- `pants-fire` — Absurdly false, ridiculous claims

## Notes/Quirks
- **No header row:** TSV files must be read without headers; assign column names programmatically when loading
- **Label mapping required:** 6-class labels will need mapping to 3-class scheme (factual / mixed / false)
- **Tab-separated format:** Files use `.tsv` (tab-separated values), not comma-separated
- **Numeric counts:** Count columns are stored as numeric integers
- **Speaker/party metadata:** Not all rows have complete metadata; sparse for non-US figures
- **Subject categories:** Multiple topics per statement, comma-separated
- **Context field varies:** Mix of URLs, quotes, event descriptions, media references
- **Special characters:** Minimal; mostly clean ASCII; rare Unicode

## License/Terms
The LIAR dataset is provided for research use. Citation of Wang (2017) is required. See original repository for detailed terms.

## Reproduction Steps

### PowerShell (Windows)
```powershell
$liarUrl = "https://www.cs.ucsb.edu/~william/data/liar_dataset.zip"
Invoke-WebRequest -Uri $liarUrl -OutFile "data\raw\liar\liar.zip"
mkdir data\raw\liar\original -Force
Copy-Item data\raw\liar\liar.zip data\raw\liar\original\
certutil -hashfile "data\raw\liar\original\liar.zip" SHA256 > data\raw\liar\original\liar_SHA256.txt
tar -xf data\raw\liar\liar.zip -C data\raw\liar
```

### Bash/Linux/macOS
```bash
liarUrl="https://www.cs.ucsb.edu/~william/data/liar_dataset.zip"
wget -O data/raw/liar/liar.zip "$liarUrl"
mkdir -p data/raw/liar/original
cp data/raw/liar/liar.zip data/raw/liar/original/
sha256sum data/raw/liar/liar.zip > data/raw/liar/original/liar_SHA256.txt
unzip -d data/raw/liar data/raw/liar/liar.zip
```

## Example Rows (Sanitized)

**From test.tsv:**
```
11972.json	true	Building a wall on the U.S.-Mexico border will take literally years.	immigration	rick-perry	Governor	Texas	republican	30	30	42	23	18	Radio interview
```

**From train.tsv:**
```
2635.json	false	Says the Annies List political group supports third-trimester abortions on demand.	abortion	dwayne-bohac	State representative	Texas	republican	0	1	0	0	0	a mailer
```

**From valid.tsv:**
```
12134.json	barely-true	We have less Americans working now than in the 70s.	economy,jobs	vicky-hartzler	U.S. Representative	Missouri	republican	1	0	1	0	0	an interview with ABC17 News
```