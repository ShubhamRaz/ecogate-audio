# Problem Statement (`docs/research/problem_statement.md`)

## 1. Ecological and Agricultural Context
Avian pest monitoring and biodiversity assessment in smart rice farming environments require continuous acoustic sensing across remote field sites ([Mendeley Dataset DOI: 10.17632/p6m5tmr477.1](https://data.mendeley.com/datasets/p6m5tmr477)). Acoustic sensors such as AudioMoth and MEMS microphone arrays record thousands of short (2–3 second) audio segments daily containing:
1. Target focal bird species of agricultural or ecological interest (*Common waxbill*, *Red-billed quelea*, *Village weaver*, *Yellow-fronted canary*).
2. Non-focal or unfamiliar bird species vocalizations (`other_birds`).
3. Complex environmental background noise (`environmental_noise`: wind, rainfall, insects, farm machinery, human speech).
4. Corrupted, clipped, or low-signal-to-noise ratio (SNR) recordings.

## 2. The Selective Inference Dilemma
Deploying high-capacity acoustic transformers or multi-scale spectrogram models ([Gong et al., 2021](https://www.isca-archive.org/interspeech_2021/gong21b_interspeech.html); [Koutini et al., 2022](https://arxiv.org/abs/2110.05069)) on every incoming clip incurs substantial computational latency and energy overhead. Conversely, relying solely on an ultra-lightweight student CNN degrades accuracy on acoustically ambiguous clips and risks false known-species assignments on unfamiliar bird vocalizations.

## 3. Formal Objective
Given an input audio clip $x$, an ultra-lightweight student model $f_s(x)$, and a stronger expert model $f_e(x)$ with invocation cost $C_{\text{expert}}$, our objective is to design a **pre-invocation routing policy** $\pi(z)$ (where $z$ comprises strictly pre-expert features extracted from $x$ and $f_s(x)$) that minimizes expected decision loss $\mathbb{E}[\mathcal{L}(\hat{y}_\pi(x), y)]$ subject to an average computational or energy budget $\mathbb{E}[E_\pi] \le B$, while distinguishing four distinct semantic outcomes:
- `known_species`
- `bird_unfamiliar`
- `nonbird`
- `unresolved`
