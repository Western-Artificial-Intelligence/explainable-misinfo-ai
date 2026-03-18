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

