# Vocal Isolation with SpeechBrain

This project explores **music source separation** for vocal isolation using **Conv-TasNet** and **SepFormer** built with the **SpeechBrain** toolkit.

## Project Overview
The repository contains a notebook-driven course project and a refactored training layout for publishing on GitHub. The workflow covers:
- MUSDB18 data preparation
- metadata CSV creation
- model training with SpeechBrain
- evaluation and qualitative listening

## Repository Structure
```text
.
├── configs/
│   ├── convtasnet.yaml
│   ├── sepformer.yaml
│   └── *_original.yaml
├── notebooks/
│   └── Project.ipynb
├── scripts/
│   └── prepare_musdb.py
├── src/
│   ├── train.py
│   └── train_original.py
├── data/
│   ├── raw/
│   └── processed/
├── results/
├── requirements.txt
└── README.md
```

## Installation
```bash
pip install -r requirements.txt
```

## Data Preparation
1. Download the MUSDB18 dataset.
2. Place the raw dataset under `data/raw/`.
3. Generate `musdb_train.csv`, `musdb_valid.csv`, and `musdb_test.csv` from the preprocessing logic in `notebooks/Project.ipynb`.

Starter command:
```bash
python scripts/prepare_musdb.py
```

## Training
Train Conv-TasNet:
```bash
python src/train.py configs/convtasnet.yaml
```

Train SepFormer:
```bash
python src/train.py configs/sepformer.yaml
```

## Notes
- The original uploaded files are preserved as `*_original.yaml` and `train_original.py`.
- The refactored files are intended to make the repository easier to read and maintain.
- You may still need to adapt dataset loading hooks depending on how your CSV manifests are generated.

## Acknowledgment
This project builds on the **SpeechBrain** toolkit for speech and audio processing.

SpeechBrain repository:
- SpeechBrain GitHub: https://github.com/speechbrain/speechbrain

Please cite SpeechBrain if you use this toolkit:

```bibtex
@article{speechbrain2021,
  title={SpeechBrain: A General-Purpose Speech Toolkit},
  author={Ravanelli, Mirco and others},
  journal={arXiv preprint arXiv:2106.04624},
  year={2021}
}
```
