# PFE — Qualité et agressivité chimique de l’eau par apprentissage automatique

**Sujet :** *Application de l’apprentissage automatique à l’évaluation de la qualité et de l’agressivité chimique de l’eau : cas d’étude appliqué au dessalement d’eau de mer.*

## 1. Contexte

Le dessalement par osmose inverse permet de produire une eau faiblement minéralisée qui doit ensuite être contrôlée et, si nécessaire, reminéralisée avant distribution. Ce projet étudie deux dimensions complémentaires :

- la **classification de la potabilité** à partir de paramètres physico-chimiques ;
- l’**évaluation de l’agressivité/corrosivité chimique** à partir de l’indice de Larson, puis sa modélisation par apprentissage automatique.

La station de dessalement d’Al Hoceima constitue le **contexte d’application et de discussion** du PFE. Les jeux de données fournis ne sont pas présentés comme des mesures directes de cette station.

## 2. Problématique

> Dans quelle mesure les méthodes d’apprentissage automatique peuvent-elles exploiter les paramètres physico-chimiques de l’eau afin d’évaluer sa potabilité et d’anticiper son caractère agressif/corrosif dans un contexte de dessalement d’eau de mer ?

## 3. Objectifs

1. Explorer et contrôler la qualité des deux jeux de données.
2. Construire un pipeline reproductible de classification de la potabilité.
3. Calculer l’indice de Larson avec conversion correcte des concentrations en meq/L.
4. Comparer plusieurs modèles de classification avec une baseline naïve.
5. Évaluer les modèles avec plusieurs métriques adaptées aux classes déséquilibrées.
6. Étudier l’agressivité selon deux expériences :
   - **Expérience A** : avec `Cl`, `SO4`, `HCO3` — reproduction prédictive de la règle de Larson ;
   - **Expérience B** : sans `Cl`, `SO4`, `HCO3` — prédiction indirecte à partir des autres paramètres.
7. Interpréter les modèles par permutation importance et, lorsque pertinent, SHAP.
8. Discuter les limites de transposition vers une station réelle de dessalement.

## 4. Jeux de données

### `water_potability.csv`

Utilisé pour la classification binaire `Potability`.

Principales variables : `ph`, `Hardness`, `Solids`, `Chloramines`, `Sulfate`, `Conductivity`, `Organic_carbon`, `Trihalomethanes`, `Turbidity`.

### `water_quality.xlsx`

Contient les feuilles `MHLATHUZE` et `LUVUVU` avec des paramètres physico-chimiques tels que `pH`, `EC`, `TDS`, `Na`, `K`, `Ca`, `Mg`, `Cl`, `HCO3`, `SO4`, `NO3`.

La colonne `Y` n’est **pas utilisée**, car sa signification n’est pas documentée dans les fichiers fournis.

## 5. Pipeline méthodologique

```text
Données brutes
    │
    ├── Potabilité
    │     ├── EDA et contrôle qualité
    │     ├── Split train/test stratifié
    │     ├── Imputation dans le Pipeline
    │     ├── Standardisation si nécessaire
    │     ├── Baseline + modèles ML
    │     ├── Validation croisée
    │     ├── Test final
    │     └── Interprétabilité
    │
    └── Agressivité
          ├── EDA et contrôle qualité
          ├── Conversion mg/L → meq/L
          ├── Calcul de Larson
          ├── Expérience A : avec Cl/SO4/HCO3
          ├── Expérience B : sans Cl/SO4/HCO3
          ├── Validation croisée + test final
          └── Comparaison scientifique des deux expériences
```

## 6. Structure du projet

```text
pfe-water-quality-ml/
├── data/
│   ├── raw/
│   │   ├── water_potability.csv
│   │   └── water_quality.xlsx
│   └── processed/
├── notebooks/
│   ├── 01_exploration_potabilite.ipynb
│   ├── 02_exploration_agressivite.ipynb
│   ├── 03_modelisation_potabilite.ipynb
│   ├── 04_modelisation_agressivite.ipynb
│   └── 05_synthese_discussion.ipynb
├── src/
│   ├── data_cleaning.py
│   ├── indices_chimiques.py
│   ├── preprocessing.py
│   ├── modeling.py
│   ├── evaluation.py
│   └── interpretability.py
├── tests/
│   └── test_indices_chimiques.py
├── models/
├── reports/
│   ├── figures/
│   └── results/
├── requirements.txt
└── README.md
```

Les fichiers de `data/raw/` ne sont jamais modifiés. Les données préparées, résultats, figures et modèles sont écrits respectivement dans `data/processed/`, `reports/` et `models/`.

## 7. Modèles comparés

- `DummyClassifier` — baseline naïve ;
- Régression logistique ;
- Random Forest ;
- SVM à noyau RBF ;
- XGBoost, si la bibliothèque est disponible.

## 8. Métriques

Les performances sont comparées avec :

- Accuracy ;
- Precision ;
- Recall ;
- F1-score ;
- ROC-AUC ;
- PR-AUC / Average Precision ;
- MCC (Matthews Correlation Coefficient) ;
- matrice de confusion brute et normalisée.

Le choix du meilleur modèle ne repose pas uniquement sur l’accuracy. Le F1, le ROC-AUC, le PR-AUC et le MCC sont considérés conjointement.

## 9. Remarques scientifiques importantes

- L’imputation et la standardisation sont effectuées **à l’intérieur des pipelines scikit-learn** afin de limiter les fuites d’information lors de la validation croisée.
- L’indice de Larson est calculé après conversion des ions `Cl`, `SO4` et `HCO3` de mg/L vers meq/L.
- Une performance très élevée dans l’**Expérience A** n’est pas interprétée comme une découverte indépendante du ML, car la cible Larson dépend directement de `Cl`, `SO4` et `HCO3`.
- L’**Expérience B** retire ces trois variables pour tester une capacité de prédiction indirecte plus intéressante scientifiquement.
- L’indice de Langelier est documenté dans le code, mais n’est pas calculé si la température nécessaire n’est pas disponible : aucune valeur arbitraire n’est inventée.
- Les données publiques utilisées ici ne sont pas présentées comme des mesures directes de la station d’Al Hoceima.

### Séparation sélection / test final

Pour éviter tout biais méthodologique, les modèles sont comparés et sélectionnés **uniquement par validation croisée stratifiée sur le jeu d’entraînement**. Le jeu de test n’est consulté qu’après cette sélection, afin de fournir une estimation finale indépendante de la généralisation du modèle retenu. Cette règle est appliquée à la potabilité ainsi qu’aux expériences A et B sur l’agressivité.

## 10. Installation

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt
jupyter notebook
```

## 11. Ordre d’exécution

Exécuter les notebooks dans cet ordre :

1. `01_exploration_potabilite.ipynb`
2. `02_exploration_agressivite.ipynb`
3. `03_modelisation_potabilite.ipynb`
4. `04_modelisation_agressivite.ipynb`
5. `05_synthese_discussion.ipynb`

## 12. Tests

```bash
pytest -q
```

Les tests vérifient notamment la conversion mg/L → meq/L, les seuils de classification de Larson et la gestion des valeurs invalides.

## 13. Résultats produits

Après exécution des notebooks :

- `data/processed/` : données nettoyées et données d’agressivité enrichies ;
- `reports/results/` : métriques CSV, comparaisons de modèles et importance des variables ;
- `reports/figures/` : matrices de confusion, courbes ROC/PR et figures utiles au mémoire ;
- `models/` : meilleurs pipelines sérialisés avec `joblib`.

## 14. Résultats de référence obtenus

Lors de l’exécution de contrôle fournie avec cette version du projet (`random_state=42`), les meilleurs modèles non naïfs donnent :

| Axe | Modèle | F1 test | ROC-AUC | PR-AUC | MCC |
|---|---|---:|---:|---:|---:|
| Potabilité | SVM-RBF | 0.512 | 0.644 | 0.566 | 0.203 |
| Agressivité A — avec `Cl/SO4/HCO3` | XGBoost | 0.935 | 0.963 | 0.968 | 0.861 |
| Agressivité B — sans `Cl/SO4/HCO3` | XGBoost | 0.807 | 0.859 | 0.860 | 0.619 |

Ces valeurs servent de **résultats de référence reproductibles dans l’environnement de contrôle**. Elles peuvent varier légèrement selon les versions des bibliothèques. Le recul entre A et B est un résultat important : il confirme qu’une partie de la performance de A provient de l’accès direct aux variables qui définissent Larson, tandis que B conserve néanmoins un signal prédictif indirect.

## 15. Limites

Ce projet démontre une méthodologie de Data Science applicable à la surveillance de la qualité chimique de l’eau. Une validation industrielle à Al Hoceima nécessiterait cependant des données locales horodatées, une documentation complète des capteurs et analyses de laboratoire, ainsi qu’un protocole de validation externe.
