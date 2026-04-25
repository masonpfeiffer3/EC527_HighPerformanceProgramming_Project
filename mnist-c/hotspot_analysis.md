# MNIST Serial Model – Memory & FLOP Analysis

## Model Architecture

| Layer | Neurons | Weight Matrix | Size (floats) |
|---|---|---|---|
| Input (IN) | 784 | — | — |
| Hidden 0 (H0) | 16 | H0_W [16×784] | 12,544 |
| Hidden 1 (H1) | 16 | H1_W [16×16] | 256 |
| Output (OUT) | 10 | L_W [10×16] | 160 |

> All values are per-call counts of 32-bit float read/write accesses and arithmetic FLOPs.
> Sigmoid FLOPs counted as 4 ops (neg + exp + add + div). Sigmoid′ counted as 10 ops (2× sigmoid + sub + mul).

---

## init_MNIST — Called Once at Startup

| Phase | Reads | Writes | FLOPs |
|---|---|---|---|
| calloc zeros (all arrays + all matrices) | 0 | 26,984 | 0 |
| init_matrix_rand overwrites weight matrices | 0 | 12,960 | 51,840 |
| **Total** | **0** | **39,944** | **51,840** |

> Not a hotspot. Called exactly once. The dominant term (H0_W + H0_W_grad calloc: 2×12,544 = 25,088 writes)
> is negligible relative to training. FLOPs are double-precision arithmetic inside `fRand`.

---

## test_MNIST — Called Once, 10,000 Samples

| Operation | Reads | Writes | FLOPs |
|---|---|---|---|
| copyImageToInput | 784 | 784 | 0 |
| feedforward (all layers) | 26,130 | 294 | 26,130 |
| vector_max (OUT) | 10 | 0 | 0 |
| **Per sample** | **26,924** | **1,078** | **26,130** |
| **× 10,000 samples** | **269,240,000** | **10,780,000** | **261,300,000** |

> Not a hotspot. Forward pass only — no backprop, no gradient accumulation.
> Total memory accesses: ~280M.

---

## train_MNIST — Per Epoch: 60,000 Samples / 6,000 Batches

### Per Sample (× 60,000)

#### copyImageToInput

| Call | Reads | Writes | FLOPs |
|---|---|---|---|
| Copy image pixels into IN [784] | 784 | 784 | 0 |

#### feedforward — IN → H0

| Call | Reads | Writes | FLOPs |
|---|---|---|---|
| new_array ×2 (calloc) | 0 | 32 | 0 |
| **matrix_vector_mult  H0_W × IN  [16×784]×[784]→[16]** | **25,088** | **16** | **25,088** |
| vector_vector_add  H0_B + mvm_res | 32 | 16 | 16 |
| vector_copy  z → H0_Z | 16 | 16 | 0 |
| sigmoid_arr  z [16] | 16 | 16 | 64 |
| vector_copy  z → H0 | 16 | 16 | 0 |
| **Subtotal** | **25,168** | **112** | **25,168** |

#### feedforward — H0 → H1

| Call | Reads | Writes | FLOPs |
|---|---|---|---|
| new_array ×2 (calloc) | 0 | 32 | 0 |
| matrix_vector_mult  H1_W × H0  [16×16]×[16]→[16] | 512 | 16 | 512 |
| vector_vector_add  H1_B + mvm_res | 32 | 16 | 16 |
| vector_copy  z → H1_Z | 16 | 16 | 0 |
| sigmoid_arr  z [16] | 16 | 16 | 64 |
| vector_copy  z → H1 | 16 | 16 | 0 |
| **Subtotal** | **592** | **112** | **592** |

#### feedforward — H1 → OUT

| Call | Reads | Writes | FLOPs |
|---|---|---|---|
| new_array ×2 (calloc) | 0 | 20 | 0 |
| matrix_vector_mult  L_W × H1  [10×16]×[16]→[10] | 320 | 10 | 320 |
| vector_vector_add  L_B + mvm_res | 20 | 10 | 10 |
| vector_copy  z → L_Z | 10 | 10 | 0 |
| sigmoid_arr  z [10] | 10 | 10 | 40 |
| vector_copy  z → OUT | 10 | 10 | 0 |
| **Subtotal** | **370** | **70** | **370** |

---

#### backprop — Output Layer

| Call | Reads | Writes | FLOPs |
|---|---|---|---|
| new_array delCdelA (calloc) | 0 | 10 | 0 |
| numToVec  y [10] (calloc + set) | 0 | 11 | 0 |
| vector_vector_sub  OUT − y | 20 | 10 | 10 |
| new_array delAdelZ (calloc) | 0 | 10 | 0 |
| vector_copy  L_Z → delAdelZ | 10 | 10 | 0 |
| sigmoid_prime_arr  [10] | 10 | 10 | 100 |
| vector_vector_elementwise_mult  [10] | 20 | 10 | 10 |
| vector_vector_mult  L_B_grad ⊗ H1  [10]⊗[16]→[10×16] | 320 | 160 | 160 |
| new_matrix W_transpose (calloc) [16×10] | 0 | 160 | 0 |
| matrix_transpose  L_W  [10×16]→[16×10] | 160 | 160 | 0 |
| matrix_vector_mult  W_T × L_B_grad  [16×10]×[10]→[16] | 320 | 16 | 320 |
| **Subtotal** | **860** | **567** | **600** |

#### backprop — Hidden Layer 1

| Call | Reads | Writes | FLOPs |
|---|---|---|---|
| new_array delCdelA (calloc) | 0 | 16 | 0 |
| vector_copy  H1_A_grad → delCdelA | 16 | 16 | 0 |
| new_array delAdelZ (calloc) | 0 | 16 | 0 |
| vector_copy  H1_Z → delAdelZ | 16 | 16 | 0 |
| sigmoid_prime_arr  [16] | 16 | 16 | 160 |
| vector_vector_elementwise_mult  [16] | 32 | 16 | 16 |
| vector_vector_mult  H1_B_grad ⊗ H0  [16]⊗[16]→[16×16] | 512 | 256 | 256 |
| new_matrix W_transpose (calloc) [16×16] | 0 | 256 | 0 |
| matrix_transpose  H1_W  [16×16]→[16×16] | 256 | 256 | 0 |
| matrix_vector_mult  W_T × H1_B_grad  [16×16]×[16]→[16] | 512 | 16 | 512 |
| **Subtotal** | **1,360** | **880** | **944** |

#### backprop — Hidden Layer 0

| Call | Reads | Writes | FLOPs |
|---|---|---|---|
| new_array delCdelA (calloc) | 0 | 16 | 0 |
| vector_copy  H0_A_grad → delCdelA | 16 | 16 | 0 |
| new_array delAdelZ (calloc) | 0 | 16 | 0 |
| vector_copy  H0_Z → delAdelZ | 16 | 16 | 0 |
| sigmoid_prime_arr  [16] | 16 | 16 | 160 |
| vector_vector_elementwise_mult  [16] | 32 | 16 | 16 |
| **vector_vector_mult  H0_B_grad ⊗ IN  [16]⊗[784]→[16×784]** | **25,088** | **12,544** | **12,544** |
| **Subtotal** | **25,168** | **12,640** | **12,720** |

---

#### Gradient Accumulation

| Call | Reads | Writes | FLOPs |
|---|---|---|---|
| **matrix_matrix_add  H0_W_grad_sum += H0_W_grad  [16×784]** | **25,088** | **12,544** | **12,544** |
| matrix_matrix_add  H1_W_grad_sum += H1_W_grad  [16×16] | 512 | 256 | 256 |
| matrix_matrix_add  L_W_grad_sum  += L_W_grad   [10×16] | 320 | 160 | 160 |
| vector_vector_add  H0_B_grad_sum += H0_B_grad  [16] | 32 | 16 | 16 |
| vector_vector_add  H1_B_grad_sum += H1_B_grad  [16] | 32 | 16 | 16 |
| vector_vector_add  L_B_grad_sum  += L_B_grad   [10] | 20 | 10 | 10 |
| **Subtotal** | **26,004** | **13,002** | **13,002** |

---

#### Per-Sample Summary

| Phase | Reads | Writes | FLOPs |
|---|---|---|---|
| copyImageToInput | 784 | 784 | 0 |
| feedforward | 26,130 | 294 | 26,130 |
| backprop | 27,388 | 14,087 | 14,264 |
| gradient accumulation | 26,004 | 13,002 | 13,002 |
| **Per-sample total** | **80,306** | **28,167** | **53,396** |
| **× 60,000 samples** | **4,818,360,000** | **1,690,020,000** | **3,203,760,000** |

---

### Per Batch (× 6,000)

| Phase | Reads | Writes | FLOPs |
|---|---|---|---|
| matrix_scalar_mult ×3  H0_W_grad_sum [16×784] | 37,632 | 37,632 | 37,632 |
| matrix_scalar_mult ×3  H1_W_grad_sum [16×16] | 768 | 768 | 768 |
| matrix_scalar_mult ×3  L_W_grad_sum  [10×16] | 480 | 480 | 480 |
| vector_scalar_mult ×3  bias grad sums [16,16,10] | 126 | 126 | 126 |
| matrix_matrix_add  weight update (all layers) | 26,004 | 13,002 | 13,002 |
| vector_vector_add  bias update (all layers) | 84 | 42 | 42 |
| zero_matrix / zero_array  grad sums | 0 | 13,002 | 0 |
| **Per-batch total** | **65,094** | **65,052** | **52,050** |
| **× 6,000 batches** | **390,564,000** | **390,312,000** | **312,300,000** |

---

### train_MNIST Epoch Total

| | Reads | Writes | Total Accesses | FLOPs |
|---|---|---|---|---|
| Per-sample work ×60,000 | 4,818,360,000 | 1,690,020,000 | 6,508,380,000 | 3,203,760,000 |
| Per-batch work ×6,000 | 390,564,000 | 390,312,000 | 780,876,000 | 312,300,000 |
| **Epoch total** | **5,208,924,000** | **2,080,332,000** | **7,289,256,000** | **3,516,060,000** |

---

## Identified Kernel Hotspots

All three hotspots are driven by the same term: `n_H0 × n_in = 16 × 784 = 12,544`.

### Hotspot 1 — feedforward: Matrix-Vector Multiply

```
z_0 = H0_W * IN       [16×784] * [784] → [16]
```

| Reads | Writes | FLOPs |
|---|---|---|
| 25,088 | 16 | 25,088 |

H0_W (12,544 floats) is read once. IN (784 floats) is re-read once per output row (16 times).

---

### Hotspot 2 — backprop: Outer Product

```
H0_W_grad = delta_0 (x) IN       [16] ⊗ [784] → [16×784]
```

| Reads | Writes | FLOPs |
|---|---|---|
| 25,088 | 12,544 | 12,544 |

delta_0 (16 floats) is re-read 784 times. IN (784 floats) is re-read 16 times. Produces the largest single write in the model per sample.

---

### Hotspot 3 — gradient accumulation: Matrix Add

```
H0_W_grad_sum += H0_W_grad       [16×784] + [16×784] → [16×784]
```

| Reads | Writes | FLOPs |
|---|---|---|
| 25,088 | 12,544 | 12,544 |

Two full reads of H0_W_grad_sum and H0_W_grad, one full write back.

---

### Combined Hotspot Share (per epoch)

| | Hotspots combined | Epoch total | Share |
|---|---|---|---|
| Reads | 3 × 25,088 × 60,000 = 4,515,840,000 | 5,208,924,000 | **86.7%** |
| Writes | (12,544+12,544) × 60,000 = 1,505,280,000 | 2,080,332,000 | **72.4%** |
| FLOPs | (25,088+12,544+12,544) × 60,000 = 3,010,560,000 | 3,516,060,000 | **85.6%** |

---

## Function Cost Comparison

| Function | Total Reads | Total Writes | Total Accesses | FLOPs |
|---|---|---|---|---|
| init_MNIST (×1) | 0 | 39,944 | 39,944 | 51,840 |
| test_MNIST (×1) | 269,240,000 | 10,780,000 | 280,020,000 | 261,300,000 |
| train_MNIST (×1 epoch) | 5,208,924,000 | 2,080,332,000 | 7,289,256,000 | 3,516,060,000 |
| **train / test ratio** | **19.4×** | **193×** | **26×** | **13.5×** |
| **train / init ratio** | **—** | **52,100×** | **182,500×** | **67,800×** |
