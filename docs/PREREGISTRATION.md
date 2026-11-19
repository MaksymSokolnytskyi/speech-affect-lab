# Pre-registered success criteria

These criteria were written down and committed **before** the first run of each stage, in
the project's original (private) working history. They are reproduced here verbatim in
meaning; the verdicts in the README are computed by `experiments/make_report.py` directly
from the run logs. A criterion that is not met is reported as a result, not hidden.

| Stage | Criterion |
|---|---|
| 1. Emotion probe | Speaker-independent UAR is above chance (1/6), and the lower bound of its 95% interval is above 1/6. |
| 1. Leakage | Leakage is shown if the lower bound of the 95% paired interval of UAR(random split) − UAR(speaker split) is above 0. |
| 2. Telephone | The drop on every telephone condition is reported with a paired interval, whatever it is. |
| 2. Augmentation | Augmentation helps if the lower bound of the paired interval of UAR(B, augmented) − UAR(A, clean) is above 0 on the telephone conditions. |
| 3. Export | ONNX ≡ PyTorch: maximum absolute difference of the embeddings ≤ 1e-4 (before quantisation). |
| 3. int8 | int8 is acceptable if the lower bound of the paired interval of UAR(int8) − UAR(fp32) is ≥ −1 pp **and** p50 latency drops by at least 1.5×. |
| Added before its run | Classical baseline: eGeMAPSv02 (88 openSMILE functionals) → the same StandardScaler + logistic regression, the same nested speaker-grouped CV (only C is tuned), the same folds as stage 1. WavLM beats the classical features if the lower bound of the paired interval of UAR(WavLM) − UAR(eGeMAPS) is above 0. |

**Statistics.** UAR (balanced accuracy) is the primary metric. Intervals are 95% percentile
bootstrap intervals over *speakers* (1,000 resamples); comparisons on the same clips use a
paired bootstrap of the difference.

**Exploratory, no criteria.** Representation analyses (layer-wise CKA between WavLM and
HuBERT, Grassmann distances between emotion subspaces, variance decomposition by speaker /
emotion / sentence, Fisher criteria) are descriptive and were not validated on held-out data
beyond the split-half speaker control reported in the paper.
