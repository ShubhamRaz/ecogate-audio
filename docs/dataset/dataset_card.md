# Dataset Card (`docs/dataset/dataset_card.md`)

- **Dataset Title**: [Multimodal Computer Vision and Acoustic Bird Detection Dataset for Smart Rice Farming Environments](https://data.mendeley.com/datasets/p6m5tmr477)
- **DOI**: `10.17632/p6m5tmr477.1`
- **Declared License**: CC BY 4.0 (Note: `environmental_noise` files matching [ESC-50](https://github.com/karolpiczak/esc-50) filenames must have their original Creative Commons licenses verified before standalone redistribution).
- **Collection Context**: Agricultural rice farming ecosystems in Kenya and Rwanda using AudioMoth and MEMS acoustic sensors (~2–3 second clips).
- **Acoustic Categories (6)**:
  1. `common_waxbill` (*Estrilda astrild*) — Known focal species (`known_species`)
  2. `red_billed_quelea` (*Quelea quelea*) — Known focal species (`known_species`)
  3. `village_weaver` (*Ploceus cucullatus*) — Known focal species (`known_species`)
  4. `yellow_fronted_canary` (*Crithagra mozambica*) — Known focal species (`known_species`)
  5. `other_birds` — Auxiliary bird vocalizations mapped to `bird_unfamiliar`
  6. `environmental_noise` — Wind, rain, insects, farm machinery mapped to `nonbird`
- **Modality Policy**: Strictly **Audio-First**. The 763 JPEG images (`3,875` bounding boxes of class `Bird`) in the archive are unsynchronized with the audio clips and are excluded from acoustic training and evaluation.
