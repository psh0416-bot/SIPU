# Dependencies

## Recommended (exact artifact environment)

Use:

```bash
conda env create -f environment.pytorch_env.yml
conda activate pytorch_env
```

The environment file is intentionally minimal and avoids machine-specific metadata.

## Core versions used

- Python 3.10.16
- torch 2.6.0+cu124
- torch-geometric 2.6.0
- numpy 2.2.6
- pandas 2.3.3
- scikit-learn 1.7.2
- scipy 1.15.3

## Notes

- If you do not use conda, install PyTorch/PyG wheels compatible with your CUDA/CPU target first, then run `pip install -r requirements.txt`.
