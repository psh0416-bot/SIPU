# Reproducibility Package

This folder is prepared for paper artifact submission.

## 1) Directory layout

- `src/`: training/evaluation code
- `data/`: datasets used in submitted experiments (cached tensors)
- `environment.pytorch_env.yml`: exported conda environment
- `requirements.txt`: minimal pip requirements

## 2) Environment setup

Recommended:

```bash
conda env create -f environment.pytorch_env.yml
conda activate pytorch_env
```

Fallback (manual):

```bash
pip install -r requirements.txt
```

## 3) Important run rule

Run experiment entrypoints **from `src/`**.  
`src/data.py` uses relative paths (`../data`, `../out`).

## 4) Example commands

Smoke test:

```bash
cd src
python -s main.py --data chameleon-filtered --model signedpu --loss sbre-lsp --seed 0 --study main --patience 5
```

Main experiment (SignedPU):

```bash
cd src
bash signedpu_main.sh
```

Main baseline suite:

```bash
bash baseline.sh
```

Main-unknown baseline suite:

```bash
cd src
bash baseline_unknown.sh
```

## 5) Datasets used by these scripts

- chameleon-filtered
- twitch-en
- facebook
- pubmed
- actor
- roman-empire
- amazon-computers
- amazon-photo

We provide our pre-processed hetero-PU datasets [here](https://drive.google.com/drive/folders/1B0EV_L8PyUDuXMHQVF0UKZMC4ETXKXNN?usp=drive_link).

## 6) Outputs

Outputs are written under `out/` (created automatically), mainly:

- `out/log/main`
- `out/log/main_unknown`
