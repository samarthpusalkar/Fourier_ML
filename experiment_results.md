# Causal Continuous Fourier LM: Experimental Results Log

This document records the training runs and benchmarks for the Continuous Fourier Language Model (`ContinuousFourierLM`).

---

## Run 1: Colab Single-Chip TPU (LR: 2e-4) — Stable Baseline
* **Date:** June 14, 2026
* **Environment:** Google Colab
* **Hardware:** Single-chip TPU v2/v3 (1 core active / single engine)
* **Model Size:** ~120M parameters
* **Optimizer:** `Adafactor` (`relative_step=False`, `scale_parameter=False`, `warmup_init=False`)
* **Learning Rate (LR):** `2e-4` (Cosine schedule, warmup ratio: `0.05`)
* **Dataset:** FineWeb-Edu (10B Tokens stream)
* **Sequence Length:** 512 tokens
* **Per-Device Batch Size:** 8 (Effective batch size = 8)

### Run 1 Metrics
| Step | Training Loss | Validation Loss | Train Perplexity (PPL) | Val Perplexity (PPL) | Validation Accuracy |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **1000** | 6.5510 | 6.5292 | 699.96 | 684.88 | 12.68% |
| **2000** | 6.0722 | 6.1332 | 433.64 | 460.91 | 15.02% |
| **3000** | 5.6438 | 5.6944 | 282.55 | 297.19 | 17.89% |
| **4000** | 5.3843 | 5.4440 | 217.95 | 231.36 | 18.91% |
| **5000** | 5.1527 | 5.2809 | 172.90 | 196.55 | 20.18% |
| **6000** | 5.1033 | 5.1295 | 164.57 | 168.94 | 21.27% |
| **7000** | 4.9776 | 5.0389 | 145.13 | 154.30 | 21.72% |
| **8000** | 4.8470 | 4.9246 | 127.35 | 137.64 | 22.56% |
| **9000** | 4.8234 | 4.8471 | 124.38 | 127.37 | 23.06% |
| **10000** | 4.7560 | 4.8040 | 116.28 | 122.00 | 23.35% |
| **11000** | 4.6059 | 4.7624 | 100.07 | 117.02 | 23.64% |
| **12000** | 4.7119 | 4.7042 | 111.26 | 110.41 | 24.23% |
| **13000** | 4.6033 | 4.6481 | 99.81 | 104.39 | 24.62% |

---

## Run 2: Colab Single-Chip TPU (LR: 2e-2) — Stress-Test
* **Date:** June 14, 2026
* **Environment:** Google Colab
* **Hardware:** Single-chip TPU v2/v3 (1 core active / single engine)
* **Model Size:** ~120M parameters
* **Optimizer:** `Adafactor` (`relative_step=False`, `scale_parameter=False`, `warmup_init=False`)
* **Learning Rate (LR):** `2e-2` (100x larger than baseline; Cosine schedule, warmup ratio: `0.05`)
* **Dataset:** FineWeb-Edu (10B Tokens stream)
* **Sequence Length:** 512 tokens
* **Per-Device Batch Size:** 8 (Effective batch size = 8)

### Run 2 Metrics
| Step | Training Loss | Validation Loss | Train Perplexity (PPL) | Val Perplexity (PPL) | Validation Accuracy |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **1000** | 6.4726 | 6.3868 | 647.15 | 593.94 | 13.54% |
| **2000** | 6.2179 | 6.2113 | 501.65 | 498.37 | 14.31% |
| **3000** | 6.2957 | 6.0118 | 542.25 | 408.23 | 16.19% |
| **4000** | 5.9885 | 5.9241 | 398.82 | 373.94 | 16.48% |
| **5000** | 5.9116 | 5.8581 | 369.29 | 350.04 | 16.45% |
| **6000** | 5.8270 | 5.7871 | 339.34 | 326.04 | 16.12% |
| **7000** | 5.7522 | 5.6482 | 314.88 | 283.78 | 18.14% |
| **8000** | 5.6303 | 5.6060 | 278.73 | 272.04 | 17.23% |
| **9000** | 5.6950 | 5.3982 | 297.37 | 221.00 | 19.45% |

---

## Run 3: Colab Single-Chip TPU (LR: 2e-3) — Interrupted
* **Date:** June 14, 2026
* **Environment:** Google Colab
* **Hardware:** Single-chip TPU v2/v3 (1 core active / single engine)
* **Model Size:** ~120M parameters
* **Optimizer:** `Adafactor` (`relative_step=False`, `scale_parameter=False`, `warmup_init=False`)
* **Learning Rate (LR):** `2e-3` (10x larger than baseline; Cosine schedule, warmup ratio: `0.05`)
* **Dataset:** FineWeb-Edu (10B Tokens stream)
* **Sequence Length:** 512 tokens
* **Per-Device Batch Size:** 8 (Effective batch size = 8)
* **Status:** Interrupted at Step **2,422** due to a Google Colab session disconnect.

### Run 3 Metrics
| Step | Training Loss | Validation Loss | Train Perplexity (PPL) | Val Perplexity (PPL) | Validation Accuracy |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **1000** | 6.2088 | 6.2137 | 497.12 | 499.53 | 14.83% |
| **2000** | 5.8727 | 5.9717 | 355.21 | 392.16 | 15.09% |

---

## 📊 Three-Way Run Comparison (At Step 2,000)

To understand the optimal learning rate dynamics, we compare the metrics at **Step 2,000** across all three schedules:

| Run Configuration | Training Loss | Validation Loss | Train PPL | Val PPL | Validation Accuracy |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Run 1 (2e-4) — Stable Baseline** | 6.0722 | 6.1332 | 433.64 | 460.91 | 15.02% |
| **Run 2 (2e-2) — High Stress-Test** | 6.2179 | 6.2113 | 501.65 | 498.37 | 14.31% |
| **Run 3 (2e-3) — Intermediate** | **5.8727** | **5.9717** | **355.21** | **392.16** | **15.09%** |

### 📈 Convergence & Learning Rate Analysis

1. **Initial Acceleration in Run 3 (2e-3)**
   * At Step 2,000, Run 3 (`2e-3`) achieved both the lowest validation loss (`5.9717`) and the highest validation accuracy (`15.09%`).
   * This indicates that a learning rate of `2e-3` was highly optimal for early acceleration. It learned significantly faster than the `2e-4` baseline without suffering from the immediate parameter oscillations that degraded the `2e-2` run.

2. **Stability Threshold**
   * While the `2e-2` run was too aggressive (leading to training loss bumps and accuracy regressions), the `2e-3` run represents a strong middle-ground candidate. 
   * Unfortunately, the Google Colab session disconnected at step **2,422** (11 minutes 12 seconds elapsed), preventing verification of whether this rate would remain stable throughout the entire 50,000-step cosine schedule or eventually overshoot.

3. **Colab Limits & Continuation Recommendation**
   * Since the training code includes saving to Google Drive every 2,000 steps (`OUTPUT_PATH = "/content/drive/MyDrive/CausalFourierLM_Checkpoints_BiggerDataset"`), a checkpoint at step 2,000 was successfully saved.
   * **To resume the run**, execute `tpu_streaming_fourier.py` with the command-line flag:
     ```bash
     python tpu_streaming_fourier.py --load_weights "/content/drive/MyDrive/CausalFourierLM_Checkpoints_BiggerDataset/checkpoint-2000/pytorch_model.bin" --learning_rate 2e-3
     ```
     *(Adjust the path if the Trainer saved it as a safetensors file or a different naming convention).*
