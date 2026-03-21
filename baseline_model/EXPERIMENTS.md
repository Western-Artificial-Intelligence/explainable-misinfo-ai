# Experiment Tracking

## Class Weights (`domain_adversarial.py`)

| Version | false | mixed | true |
|---------|-------|-------|------|
| v1      | 1.4   | 1.7   | 1.0  |
| v2      | 1.4   | 1.7   | 1.0  |
| v3      | 2.0   | 1.7   | 1.0  |
| v4      | 1.7   | 1.7   | 1.0  |


---

## RoBERTa Two-Phase (`twophase.yaml`)

| Setting              | v1   | v2   | v3/v4 |
|----------------------|------|------|-------|
| lambda_adv           | 0.5  | 0.1  | 0.3   |
| Phase 1 epochs       | 2    | 10   | 10    |
| Phase 1 lr           | 3e-4 | 3e-4 | 3e-4  |
| Phase 1 early_stop   | 2    | 5    | 5     |
| Phase 2 epochs       | 3    | 20   | 20    |
| Phase 2 head_lr      | 3e-4 | 1e-4 | 1e-4  |
| Phase 2 lora_top_lr  | 1e-4 | 3e-5 | 3e-5  |
| Phase 2 lora_lr_decay| 0.9  | 0.9  | 0.9   |
| Phase 2 weight_decay | 0.01 | 0.01 | 0.01  |
| Phase 2 early_stop   | 2    | 7    | 7     |


---

## DistilRoBERTa Two-Phase (`twophase_distil.yaml`)

| Setting                 | v1   | v2   | v3/v4 |
|-------------------------|------|------|-------|
| lambda_adv              | 0.5  | 0.1  | 0.3   |
| Phase 1 epochs          | 2    | 10   | 10    |
| Phase 1 lr              | 3e-4 | 3e-4 | 3e-4  |
| Phase 1 early_stop      | 2    | 5    | 5     |
| Phase 2 epochs          | 3    | 20   | 20    |
| Phase 2 head_lr         | 3e-4 | 1e-4 | 1e-4  |
| Phase 2 encoder_top_lr  | 1e-4 | 2e-5 | 2e-5  |
| Phase 2 encoder_lr_decay| 0.9  | 0.9  | 0.9   |
| Phase 2 weight_decay    | 0.01 | 0.01 | 0.05  |
| Phase 2 early_stop      | 2    | 7    | 7     |


---

## Shared Settings (unchanged across all versions)

| Setting             | Value                |
|---------------------|----------------------|
| LoRA r / alpha / dropout | 8 / 16 / 0.1 (RoBERTa only) |
| LoRA target_modules | ["query", "value"]   |
| Phase 1 batch_size  | 32                   |
| Phase 2 batch_size  | 32                   |
| Phase 1 warmup_ratio| 0.06                 |
| Phase 2 warmup_ratio| 0.06                 |
| grad_clip_norm      | 1.0                  |
| seed                | 42                   |
| max_length          | 256                  |
| eval_every_steps    | 500                  |

---

## Results (Val Macro-F1)

| Model     | v1     | v2         | v3     | v4  |
|-----------|--------|------------|--------|-----|
| RoBERTa   | 0.5272 | 0.5759     | 0.5770 | 0.5738 |
| Distil    | 0.5299 | **0.5898** | 0.5729 | 0.5701 |
| RoBERTa+MLM | —   | —          | —      | TBD    |


### Per-Class F1 (best epoch)

#### RoBERTa

| Class | v1     | v2     | v3     | v4  |
|-------|--------|--------|--------|-----|
| false | 0.4800 | 0.4729 | 0.5300 | 0.4997 |
| mixed | 0.5539 | 0.5831 | 0.5287 | 0.5589 |
| true  | 0.4429 | 0.6711 | 0.6434 | 0.6595 |


#### DistilRoBERTa

| Class | v1     | v2     | v3     | v4     |
|-------|--------|--------|--------|--------|
| false | 0.4400 | 0.4497 | 0.5274 | 0.4825 |
| mixed | 0.5557 | 0.6254 | 0.5431 | 0.5082 |
| true  | 0.5939 | 0.6583 | 0.6178 | 0.6421 |


### Key Metrics

| Metric         | RoBERTa v2 | Distil v2  | RoBERTa v4 | Distil v4  |
|----------------|------------|------------|------------|------------|
| Overfit gap    | 2%         | 25%        | 1%         | 6%         |
| ECE            | 0.04       | 0.26       | 0.02       | 0.14       |
| Domain acc     | 98%        | 96%        | 98%        | 50%        |

---

## Notes

- **v1 -> v2**: Increased epochs, early stopping patience, lowered Phase 2 LRs. Fixed Phase 2 collapse.
- **v2 -> v3**: Increased false class weight (1.4->2.0), lambda_adv (0.1->0.3), distil weight_decay (0.01->0.05). False F1 improved but mixed F1 dropped.
- **v3 -> v4**: Reduced false class weight (2.0->1.7) to balance false/mixed tradeoff. Distil overfitting improved (25%->6%), ECE improved (0.26->0.14). Domain acc dropped to 50% (near chance) for distil — adversarial working well. RoBERTa ECE best yet (0.02). Overall F1 still below v2 best.

---

## Binary Classification Experiments (v5)

Pipeline converted from 3-way (false/mixed/true) to binary (false/true). All models trained on GCP `g2-standard-8` with NVIDIA L4 GPU.

### MLM Pre-training

| Setting | Value |
|---------|-------|
| Base model | roberta-base |
| Dataset | `ioverho/misinfo-general` (4.2M examples, 2017-2022) |
| Epochs | 1 |
| Max seq length | 256 |
| MLM probability | 15% |
| Batch size | 64 |
| Learning rate | 2e-5 |
| Training time | ~13.4 hours |
| Final train loss | 1.0467 |
| Final eval loss | 0.9388 |
| Output | `models/mlm/roberta-misinfo-mlm` |

---

### Results Summary (Val Macro-F1)

| Rank | Model | Backbone | Best Val Macro-F1 | Val Acc | ECE |
|------|-------|----------|-------------------|---------|------|
| 1 | Baseline RoBERTa | roberta-base | **0.7470** | 75.54% | 0.0248 |
| 2 | Baseline MLM | roberta-misinfo-mlm | 0.7468 | 75.39% | 0.0322 |
| 3 | Two-Phase MLM | roberta-misinfo-mlm | 0.7429 | 75.40% | 0.0270 |
| 4 | Two-Phase RoBERTa | roberta-base | 0.7387 | 75.26% | 0.0279 |
| 5 | Two-Phase DistilRoBERTa | distilroberta-base | 0.7338 | 74.85% | 0.0378 |

---

### Per-Class Metrics (at best epoch)

| Model | false P | false R | false F1 | true P | true R | true F1 |
|-------|---------|---------|----------|--------|--------|---------|
| Baseline RoBERTa | 0.7483 | 0.8437 | 0.7931 | 0.7673 | 0.6449 | 0.7008 |
| Baseline MLM | 0.7465 | 0.8437 | 0.7921 | 0.7664 | 0.6415 | 0.6984 |
| Two-Phase MLM | 0.7376 | 0.8650 | 0.7963 | 0.7845 | 0.6151 | 0.6895 |
| Two-Phase RoBERTa | 0.7284 | 0.8850 | 0.7991 | 0.8032 | 0.5870 | 0.6783 |
| Two-Phase DistilRoBERTa | 0.7239 | 0.8850 | 0.7964 | 0.8006 | 0.5777 | 0.6711 |

---

### Training Details

| Model | Epochs Run | Best Epoch | Early Stopped | Domain Acc |
|-------|-----------|------------|---------------|------------|
| Baseline RoBERTa | 55 | 35 | Yes (patience 20) | N/A |
| Baseline MLM | 66 | 35 | Yes (patience 20) | N/A |
| Two-Phase RoBERTa | P1: 6, P2: 20 | P2 Ep 20 | P1 only (patience 5) | 98.30% |
| Two-Phase MLM | P1: 7, P2: 20 | P2 Ep 14 | P1 only (patience 5) | 98.42% |
| Two-Phase DistilRoBERTa | P1: 7, P2: 10 | P2 Ep 3 | Both (P1: 5, P2: 7) | 32.44% |

---

### v5 Observations

- All models are close in performance (F1 range: 0.7338–0.7470), a significant jump from 3-way v4 best of 0.5898.
- Baseline single-phase training slightly outperforms two-phase approaches.
- MLM pre-training on misinformation corpus provided no meaningful advantage over vanilla RoBERTa.
- All models show consistent class imbalance: high recall on "false" (0.84–0.89) but lower recall on "true" (0.55–0.65).
- DistilRoBERTa had training instability (val loss spiked to 20.97 during Phase 2) and lowest domain accuracy (32.44%), suggesting the adversarial head dominated.
- ECE (calibration) is low across all models (0.02–0.04), with Baseline RoBERTa best calibrated (0.0248).

