# PFE — Qualité et agressivité chimique de l'eau par apprentissage automatique

**Sujet :** Application de l'apprentissage automatique à l'évaluation de la qualité et de l'agressivité chimique de l'eau : cas d'étude appliqué au dessalement d'eau de mer.

## Pipeline respecté
1. Exploration de la potabilité.
2. Exploration de l'agressivité.
3. Modélisation de la potabilité.
4. Modélisation de l'agressivité.
5. Synthèse et discussion.

Les fichiers `data/raw/` ne sont jamais modifiés. Les sorties sont écrites dans `data/processed/`, `reports/`, et `models/`.

## Installation
```bash
python -m venv .venv
# Windows : .venv\Scripts\activate
# Linux/macOS : source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook
```

## Remarques scientifiques
- `water_potability.csv` sert à la classification `Potability`.
- `water_quality.xlsx` contient les feuilles `MHLATHUZE` et `LUVUVU`.
- `Y` n'est pas utilisé car sa signification n'est pas documentée dans les fichiers fournis.
- Larson est calculé à partir de `Cl`, `SO4`, `HCO3`.
- Langelier n'est pas calculé sans température : aucune valeur arbitraire n'est inventée.
- Les données fournies ne sont pas présentées comme des mesures directes d'Al Hoceima ; la station est le contexte d'application et de discussion.
