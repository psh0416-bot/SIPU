# Dataset Notes

## Included datasets (used in provided run scripts)

The following datasets are included as cached experiment-ready tensors under `data/cached/<dataset>/`:

- chameleon-filtered
- twitch-en
- facebook
- pubmed
- actor
- roman-empire
- amazon-computers
- amazon-photo

Each cached directory contains:

- `x.npy`
- `y.npy`
- `edges.npy`

## Included local raw helper file

This file is included because it is not always fetched by default dataset APIs:

- `data/npz/chameleon_filtered.npz`

## Auto-downloaded/auto-generated datasets

Other datasets (Planetoid, Amazon, Coauthor, Actor, HeterophilousGraphDataset, etc.) are loaded via PyG and cached under `data/` on first run.

## Path convention

- Code expects `data/` at package root.
- Run entrypoints from `src/` so that `../data` resolves correctly.
