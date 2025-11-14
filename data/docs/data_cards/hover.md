\# Dataset: HoVer



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

