"""Prétraitement et construction de pipelines scikit-learn."""

from collections.abc import Sequence

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def numeric_preprocessor(
    features: Sequence[str],
    scale: bool = True,
) -> ColumnTransformer:
    """Construire le préprocesseur numérique.

    L'imputation est toujours intégrée au pipeline pour éviter qu'elle soit
    apprise sur l'ensemble des données avant la validation croisée.
    """
    steps: list[tuple[str, object]] = [
        ("imputer", SimpleImputer(strategy="median")),
    ]

    if scale:
        steps.append(("scaler", StandardScaler()))

    numeric_pipeline = Pipeline(steps)

    return ColumnTransformer(
        transformers=[("num", numeric_pipeline, list(features))],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def make_model_pipeline(
    model: object,
    numeric_features: Sequence[str],
    scale: bool = True,
) -> Pipeline:
    """Assembler prétraitement et classifieur dans un pipeline unique."""
    return Pipeline(
        steps=[
            ("preprocessor", numeric_preprocessor(numeric_features, scale=scale)),
            ("model", model),
        ]
    )
