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

### Threshold Tuning 

 - baseline_roberta
Threshold   Macro-F1      Acc   true_P   true_R  true_F1  false_P  false_R false_F1
------------------------------------------------------------------------------------------
      0.10     0.5899   0.6146   0.5367   0.9680   0.6905   0.9285   0.3321   0.4892 <-- best F1
      0.15     0.6383   0.6512   0.5641   0.9455   0.7066   0.9053   0.4160   0.5700 <-- best F1
      0.20     0.6714   0.6781   0.5876   0.9231   0.7181   0.8869   0.4822   0.6248 <-- best F1
      0.25     0.6947   0.6976   0.6085   0.8947   0.7244   0.8651   0.5400   0.6650 <-- best F1
      0.30     0.7164   0.7171   0.6337   0.8607   0.7299   0.8440   0.6023   0.7029 <-- best F1
      0.35     0.7327   0.7328   0.6627   0.8114   0.7296   0.8163   0.6699   0.7359 <-- best F1
      0.40     0.7465   0.7478   0.6978   0.7623   0.7287   0.7949   0.7362   0.7644 <-- best F1
      0.45     0.7504   0.7546   0.7328   0.7043   0.7183   0.7708   0.7948   0.7826 <-- best F1
      0.50     0.7470   0.7554   0.7673   0.6449   0.7008   0.7483   0.8437   0.7931
      0.55     0.7383   0.7522   0.8021   0.5870   0.6779   0.7282   0.8842   0.7987
      0.60     0.7244   0.7451   0.8356   0.5304   0.6489   0.7095   0.9166   0.7999
      0.65     0.7069   0.7353   0.8663   0.4777   0.6159   0.6927   0.9411   0.7980
      0.70     0.6901   0.7263   0.8981   0.4328   0.5841   0.6794   0.9608   0.7960
      0.75     0.6753   0.7187   0.9281   0.3975   0.5566   0.6695   0.9754   0.7940
      0.80     0.6592   0.7094   0.9462   0.3667   0.5285   0.6602   0.9833   0.7900
      0.85     0.6432   0.6998   0.9564   0.3396   0.5012   0.6517   0.9876   0.7852
      0.90     0.6302   0.6924   0.9685   0.3178   0.4785   0.6452   0.9917   0.7818

Best macro-F1 threshold: 0.45 (F1=0.7504)
Best threshold with true recall >= 0.80: 0.35 (F1=0.7327)

- baseline_mlm
Threshold   Macro-F1      Acc   true_P   true_R  true_F1  false_P  false_R false_F1
------------------------------------------------------------------------------------------
      0.10     0.5856   0.6111   0.5344   0.9666   0.6883   0.9245   0.3269   0.4830 <-- best F1
      0.15     0.6277   0.6421   0.5573   0.9446   0.7010   0.9004   0.4004   0.5543 <-- best F1
      0.20     0.6626   0.6701   0.5810   0.9225   0.7130   0.8832   0.4684   0.6121 <-- best F1
      0.25     0.6964   0.6991   0.6102   0.8927   0.7249   0.8639   0.5443   0.6678 <-- best F1
      0.30     0.7207   0.7211   0.6389   0.8561   0.7317   0.8421   0.6132   0.7097 <-- best F1
      0.35     0.7331   0.7331   0.6636   0.8098   0.7294   0.8155   0.6719   0.7367 <-- best F1
      0.40     0.7444   0.7456   0.6950   0.7613   0.7267   0.7935   0.7330   0.7621 <-- best F1
      0.45     0.7481   0.7517   0.7239   0.7128   0.7183   0.7733   0.7827   0.7780 <-- best F1
      0.50     0.7468   0.7537   0.7534   0.6624   0.7049   0.7539   0.8267   0.7886
      0.55     0.7437   0.7548   0.7860   0.6155   0.6904   0.7381   0.8661   0.7970
      0.60     0.7341   0.7504   0.8164   0.5654   0.6681   0.7212   0.8983   0.8001
      0.65     0.7213   0.7440   0.8484   0.5159   0.6416   0.7054   0.9263   0.8009
      0.70     0.7046   0.7346   0.8768   0.4682   0.6104   0.6903   0.9474   0.7987
      0.75     0.6859   0.7235   0.8995   0.4251   0.5773   0.6768   0.9620   0.7946
      0.80     0.6675   0.7128   0.9206   0.3869   0.5448   0.6651   0.9733   0.7903
      0.85     0.6551   0.7062   0.9405   0.3615   0.5223   0.6580   0.9817   0.7879
      0.90     0.6392   0.6969   0.9524   0.3343   0.4949   0.6497   0.9867   0.7835

Best macro-F1 threshold: 0.45 (F1=0.7481)
Best threshold with true recall >= 0.80: 0.35 (F1=0.7331)

- twophase_roberta
Threshold   Macro-F1      Acc   true_P   true_R  true_F1  false_P  false_R false_F1
------------------------------------------------------------------------------------------
      0.10     0.5713   0.6012   0.5277   0.9738   0.6845   0.9354   0.3034   0.4581 <-- best F1
      0.15     0.6333   0.6474   0.5609   0.9495   0.7052   0.9096   0.4060   0.5614 <-- best F1
      0.20     0.6754   0.6813   0.5908   0.9193   0.7193   0.8839   0.4911   0.6314 <-- best F1
      0.25     0.7059   0.7076   0.6200   0.8831   0.7285   0.8587   0.5673   0.6832 <-- best F1
      0.30     0.7287   0.7287   0.6514   0.8376   0.7329   0.8318   0.6417   0.7245 <-- best F1
      0.35     0.7452   0.7458   0.6872   0.7850   0.7328   0.8061   0.7144   0.7575 <-- best F1
      0.40     0.7508   0.7541   0.7243   0.7206   0.7225   0.7776   0.7808   0.7792 <-- best F1
      0.45     0.7497   0.7576   0.7668   0.6528   0.7052   0.7520   0.8413   0.7942
      0.50     0.7387   0.7526   0.8032   0.5870   0.6783   0.7284   0.8850   0.7991
      0.55     0.7238   0.7451   0.8408   0.5258   0.6470   0.7083   0.9204   0.8006
      0.60     0.7044   0.7337   0.8689   0.4716   0.6114   0.6907   0.9431   0.7974
      0.65     0.6894   0.7258   0.8986   0.4314   0.5830   0.6790   0.9611   0.7958
      0.70     0.6791   0.7209   0.9235   0.4052   0.5633   0.6718   0.9732   0.7949
      0.75     0.6672   0.7137   0.9344   0.3824   0.5427   0.6647   0.9786   0.7916
      0.80     0.6569   0.7078   0.9452   0.3633   0.5248   0.6589   0.9832   0.7890
      0.85     0.6457   0.7012   0.9542   0.3438   0.5055   0.6530   0.9868   0.7859
      0.90     0.6335   0.6943   0.9661   0.3231   0.4843   0.6469   0.9909   0.7828

Best macro-F1 threshold: 0.40 (F1=0.7508)
Best threshold with true recall >= 0.80: 0.30 (F1=0.7287)

- twophase_mlm
Threshold   Macro-F1      Acc   true_P   true_R  true_F1  false_P  false_R false_F1
------------------------------------------------------------------------------------------
      0.10     0.5883   0.6136   0.5360   0.9698   0.6904   0.9316   0.3289   0.4862 <-- best F1
      0.15     0.6373   0.6505   0.5635   0.9467   0.7065   0.9067   0.4138   0.5682 <-- best F1
      0.20     0.6713   0.6780   0.5876   0.9229   0.7180   0.8867   0.4822   0.6247 <-- best F1
      0.25     0.6958   0.6983   0.6102   0.8881   0.7234   0.8594   0.5465   0.6682 <-- best F1
      0.30     0.7220   0.7223   0.6422   0.8462   0.7302   0.8352   0.6233   0.7138 <-- best F1
      0.35     0.7394   0.7398   0.6767   0.7931   0.7303   0.8083   0.6971   0.7486 <-- best F1
      0.40     0.7479   0.7504   0.7127   0.7337   0.7231   0.7820   0.7637   0.7727 <-- best F1
      0.45     0.7495   0.7555   0.7490   0.6761   0.7107   0.7598   0.8189   0.7883 <-- best F1
      0.50     0.7429   0.7540   0.7845   0.6151   0.6895   0.7376   0.8650   0.7963
      0.55     0.7333   0.7508   0.8253   0.5568   0.6650   0.7189   0.9058   0.8016
      0.60     0.7196   0.7436   0.8570   0.5076   0.6375   0.7032   0.9323   0.8017
      0.65     0.7035   0.7346   0.8861   0.4620   0.6074   0.6890   0.9525   0.7996
      0.70     0.6838   0.7228   0.9081   0.4183   0.5728   0.6751   0.9662   0.7949
      0.75     0.6708   0.7157   0.9294   0.3897   0.5491   0.6668   0.9763   0.7924
      0.80     0.6554   0.7070   0.9482   0.3601   0.5220   0.6581   0.9843   0.7888
      0.85     0.6418   0.6989   0.9566   0.3374   0.4989   0.6510   0.9878   0.7848
      0.90     0.6301   0.6922   0.9668   0.3180   0.4785   0.6452   0.9913   0.7816

Best macro-F1 threshold: 0.45 (F1=0.7495)
Best threshold with true recall >= 0.80: 0.30 (F1=0.7220)

- twophase_distil
Threshold   Macro-F1      Acc   true_P   true_R  true_F1  false_P  false_R false_F1
------------------------------------------------------------------------------------------
      0.10     0.6042   0.6250   0.5441   0.9612   0.6949   0.9200   0.3563   0.5136 <-- best F1
      0.15     0.6539   0.6633   0.5746   0.9318   0.7109   0.8917   0.4487   0.5970 <-- best F1
      0.20     0.6758   0.6813   0.5915   0.9136   0.7181   0.8777   0.4957   0.6336 <-- best F1
      0.25     0.6921   0.6947   0.6074   0.8841   0.7201   0.8544   0.5432   0.6641 <-- best F1
      0.30     0.7148   0.7149   0.6386   0.8251   0.7200   0.8177   0.6267   0.7096 <-- best F1
      0.35     0.7338   0.7351   0.6843   0.7494   0.7154   0.7832   0.7236   0.7522 <-- best F1
      0.40     0.7433   0.7481   0.7290   0.6890   0.7084   0.7619   0.7953   0.7782 <-- best F1
      0.45     0.7404   0.7498   0.7653   0.6300   0.6911   0.7409   0.8456   0.7898
      0.50     0.7338   0.7485   0.8006   0.5777   0.6711   0.7239   0.8850   0.7964
      0.55     0.7226   0.7439   0.8379   0.5250   0.6456   0.7076   0.9188   0.7995
      0.60     0.7088   0.7364   0.8646   0.4821   0.6190   0.6942   0.9396   0.7985
      0.65     0.6891   0.7245   0.8861   0.4358   0.5843   0.6793   0.9552   0.7940
      0.70     0.6703   0.7142   0.9145   0.3933   0.5500   0.6668   0.9706   0.7906
      0.75     0.6517   0.7033   0.9318   0.3583   0.5176   0.6562   0.9790   0.7858
      0.80     0.6407   0.6978   0.9522   0.3366   0.4974   0.6504   0.9865   0.7840
      0.85     0.6356   0.6952   0.9603   0.3273   0.4882   0.6479   0.9892   0.7830
      0.90     0.6295   0.6916   0.9644   0.3176   0.4778   0.6449   0.9906   0.7812

Best macro-F1 threshold: 0.40 (F1=0.7433)
Best threshold with true recall >= 0.80: 0.30 (F1=0.7148)

