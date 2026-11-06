# Methodology (`docs/research/methodology.md`)

## 1. Audio Preprocessing & Quality Gate
Each input recording is decoded safely, converted to mono, resampled to $32\text{ kHz}$ via polyphase filtering, RMS-normalized to $-20\text{ dBFS}$ with anti-clipping protection, and padded or center-trimmed to a $3.0\text{-second}$ window ($96,000$ samples). A 64-band Log-Mel spectrogram ($N_{\text{FFT}}=1024$, hop $=320$, $50\text{ Hz}$–$14\text{ kHz}$) and 20 MFCCs (+ $\Delta, \Delta^2$) are extracted. Clips exceeding the clipping ceiling ($>12\%$) or falling below the SNR/RMS floor are immediately routed to `unresolved` without wasting expert compute.

## 2. Two-Tier Model Hierarchy
- **Lightweight Student (`LightweightStudentCNN`)**: Compact depthwise-separable spectral-temporal convolutional network with batch normalization, SiLU activations, residual projection, a 64-D L2-normalized embedding head, a 4-class known-species head, and a binary bird-activity head.
- **Strong Expert (`StrongExpertAudioModel`)**: Higher-capacity multi-scale spectrogram network incorporating fine-grained temporal modulation ($\Delta, \Delta^2$ RMS), smoothed contextual filtering, and multi-head temporal self-attention pooling into a 128-D embedding space.

## 3. Out-of-Fold Cost-Aware Benefit Estimation
Let $\mathcal{L}(f(x_i), y_i)$ denote the composite decision loss combining known-species cross-entropy, zero-one classification error, unfamiliar-bird overconfidence penalty, and non-bird false-alarm penalty. On development folds $k \in \{1,\dots,K\}$ partitioned by duplicate/session group:
1. Train $f_s^{(-k)}$ and $f_e^{(-k)}$ on folds $\neq k$.
2. Predict on held-out fold $k$ and compute observed benefit:
   $$d_i = \mathcal{L}(f_s^{(-k)}(x_i), y_i) - \mathcal{L}(f_e^{(-k)}(x_i), y_i)$$
3. Construct strictly pre-expert routing features:
   $$z_i = [p_{\max}^{(s)}, m^{(s)}, H^{(s)}, d_{\text{proto}}^{(s)}, p_{\text{bird}}^{(s)}, \text{SNR}_i, \text{clip}_i, \text{RMS}_i, b_{\text{rem}}, H^{(s)}p_{\text{bird}}^{(s)}, (1-m^{(s)})p_{\text{bird}}^{(s)}, \mathbb{I}(\text{SNR}_i>-3)p_{\text{bird}}^{(s)}]$$
4. Fit the regularized benefit estimator $\hat{\Delta}(z) \approx \mathbb{E}[d_i \mid z]$ on $(Z_{\text{oof}}, d_{\text{oof}})$, calibrate the invocation threshold $\tau^*(\lambda, B)$ on the independent calibration split, and freeze all parameters prior to test evaluation.
