"""Fonctions d'interprétabilité des modèles."""

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance


def permutation_importance_table(
    model: object,
    X: pd.DataFrame,
    y,
    n_repeats: int = 20,
    random_state: int = 42,
    scoring: str = "f1",
) -> pd.DataFrame:
    """Calculer l'importance par permutation sur les variables originales."""
    result = permutation_importance(
        model,
        X,
        y,
        n_repeats=n_repeats,
        random_state=random_state,
        scoring=scoring,
        n_jobs=-1,
    )

    return (
        pd.DataFrame(
            {
                "feature": X.columns,
                "importance_mean": result.importances_mean,
                "importance_std": result.importances_std,
            }
        )
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )


def shap_values_for_pipeline(model_pipeline, X_sample: pd.DataFrame):
    """Calculer des valeurs SHAP pour un pipeline compatible.

    La fonction transforme d'abord les données avec le préprocesseur du pipeline,
    puis construit un explainer SHAP autour du classifieur final.
    """
    import shap

    preprocessor = model_pipeline.named_steps["preprocessor"]
    model = model_pipeline.named_steps["model"]

    transformed = preprocessor.transform(X_sample)
    feature_names = preprocessor.get_feature_names_out()

    explainer = shap.Explainer(
        model,
        transformed,
        feature_names=feature_names,
    )
    explanation = explainer(transformed)

    return explainer, explanation


def shap_global_importance(explanation) -> pd.DataFrame:
    """Résumer l'importance SHAP globale par moyenne des valeurs absolues."""
    values = np.asarray(explanation.values)
    if values.ndim == 3:
        # Classification binaire : conserver la classe positive si présente.
        values = values[:, :, -1]

    mean_abs = np.abs(values).mean(axis=0)
    feature_names = list(explanation.feature_names)

    return (
        pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
