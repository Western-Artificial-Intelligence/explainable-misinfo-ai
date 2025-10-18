\# Dataset: LIAR



\## Source

\- \*\*URL\*\*: https://www.cs.ucsb.edu/~william/data/liar\_dataset.zip

\- \*\*Date Downloaded\*\*: 2025-10-18

\- \*\*SHA256\*\*: `611c1addad919743dde15822b87a60bfb760d8f85597f25289e34621800654c7`

\- \*\*Archive Location\*\*: `data/raw/liar/original/liar.zip`



\## Citation

Wang, W. Y. (2017). "Liar, Liar Pants on Fire: A New Benchmark Dataset for Fake News Detection." \*Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (ACL)\*, pp. 422–426. \[ACL Anthology](https://aclanthology.org/P17-2067/)



\## Files Pulled

\- `train.tsv` — 10,239 rows; training set of labeled statements

\- `valid.tsv` — 1,283 rows; validation set

\- `test.tsv` — 1,266 rows; test set

\- `README` — Original dataset documentation



\## Schema (Key Files)



\### train.tsv / valid.tsv / test.tsv

Tab-separated format with the following columns:

| Column | Type | Notes |

|--------|------|-------|

| claim\_id | str | Unique identifier (filename, e.g., "2635.json") |

| label | str | Truthfulness label (6-class) |

| statement | str | The factual claim being checked |

| subject | str | Topic categories (comma-separated, e.g., "abortion") |

| speaker | str | Person making the statement |

| job\_title | str | Speaker's role/position |

| state\_info | str | Geographic context (state abbreviation) |

| party\_affiliation | str | Political party (lowercase, e.g., "republican") |

| barely\_true\_counts | str | Count of "barely true" PolitiFact labels |

| false\_counts | str | Count of "false" PolitiFact labels |

| half\_true\_counts | str | Count of "half true" PolitiFact labels |

| mostly\_true\_counts | str | Count of "mostly true" PolitiFact labels |

| pants\_on\_fire\_counts | str | Count of "pants on fire" PolitiFact labels |

| true\_counts | str | Count of "true" PolitiFact labels |

| context | str | Reference/context (URL, article excerpt, or event description) |



\*\*Row counts:\*\*

\- train.tsv: 10,239 rows

\- valid.tsv: 1,283 rows

\- test.tsv: 1,266 rows

\- \*\*Total: 12,788 rows\*\*



\## Labels (as provided)

6-class label distribution:

\- `true` — Factually accurate statements

\- `mostly-true` — Mostly accurate with minor inaccuracy

\- `half-true` — Mixed accuracy, true and false elements

\- `barely-true` — Mostly false with minor truth

\- `false` — Factually inaccurate

\- `pants-on-fire` — Absurdly false, ridiculous claims



\## Notes/Quirks

\- \*\*Label mapping required:\*\* 6-class labels will need mapping to 3-class scheme (factual / mixed / false)

\- \*\*Tab-separated format:\*\* Files use `.tsv` (tab-separated), not comma-separated

\- \*\*Numeric fields as strings:\*\* Counts are stored as strings, not integers

\- \*\*Speaker/party metadata:\*\* Not all rows have complete metadata; sparse for non-US figures

\- \*\*Subject categories:\*\* Multiple topics per statement, comma-separated

\- \*\*Context field varies:\*\* Mix of URLs, quotes, event descriptions



\## License/Terms

The LIAR dataset is provided for research use. Citation of Wang (2017) is required. See original repository for detailed terms.



\## Reproduction Steps



\### PowerShell (Windows)

```powershell

$liarUrl = "https://www.cs.ucsb.edu/~william/data/liar\_dataset.zip"

Invoke-WebRequest -Uri $liarUrl -OutFile "data\\raw\\liar\\liar.zip"

mkdir data\\raw\\liar\\original -Force

Copy-Item data\\raw\\liar\\liar.zip data\\raw\\liar\\original\\

certutil -hashfile "data\\raw\\liar\\original\\liar.zip" SHA256 > data\\raw\\liar\\original\\liar\_SHA256.txt

tar -xf data\\raw\\liar\\liar.zip -C data\\raw\\liar

```



\### Bash/Linux/macOS

```bash

liarUrl="https://www.cs.ucsb.edu/~william/data/liar\_dataset.zip"

wget -O data/raw/liar/liar.zip "$liarUrl"

mkdir -p data/raw/liar/original

cp data/raw/liar/liar.zip data/raw/liar/original/

sha256sum data/raw/liar/liar.zip > data/raw/liar/original/liar\_SHA256.txt

unzip -d data/raw/liar data/raw/liar/liar.zip

```



\## Example Row (Sanitized)



\*\*From train.tsv:\*\*

