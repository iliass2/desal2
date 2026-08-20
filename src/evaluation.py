"""Métriques, validation croisée et figures d'évaluation."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    accuracy_score,
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate


def _continuous_scores(model: object, X) -> np.ndarray | None:
    """Obtenir un score continu pour ROC-AUC et PR-AUC si disponible."""
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X)
        if probabilities.ndim == 2 and probabilities.shape[1] >= 2:
            return probabilities[:, 1]

    if hasattr(model, "decision_function"):
        return np.asarray(model.decision_function(X))

    return None


def classification_metrics(model: object, X_test, y_test) -> dict[str, float]:
    """Calculer les principales métriques binaires sur le jeu de test."""
    predictions = model.predict(X_test)
    scores = _continuous_scores(model, X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1": f1_score(y_test, predictions, zero_division=0),
        "mcc": matthews_corrcoef(y_test, predictions),
    }

    if scores is None or len(np.unique(y_test)) < 2:
        metrics["roc_auc"] = np.nan
        metrics["pr_auc"] = np.nan
    else:
        metrics["roc_auc"] = roc_auc_score(y_test, scores)
        metrics["pr_auc"] = average_precision_score(y_test, scores)

    return metrics


def cross_validation_table(
    model: object,
    X,
    y,
    n_splits: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    """Évaluer un pipeline par validation croisée stratifiée."""
    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )

    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
        "pr_auc": "average_precision",
        "mcc": "matthews_corrcoef",
    }

    scores = cross_validate(
        model,
        X,
        y,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        error_score="raise",
    )

    rows = []
    for metric in scoring:
        values = scores[f"test_{metric}"]
        rows.append(
            {
                "metric": metric,
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
            }
        )

    return pd.DataFrame(rows)


def save_confusion_matrix(
    model: object,
    X_test,
    y_test,
    title: str,
    path: str | Path,
    normalize: str | None = None,
) -> None:
    """Sauvegarder une matrice de confusion brute ou normalisée."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_estimator(
        model,
        X_test,
        y_test,
        ax=ax,
        colorbar=False,
        normalize=normalize,
        values_format=".2f" if normalize else "d",
    )
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_roc_curve(
    model: object,
    X_test,
    y_test,
    title: str,
    path: str | Path,
) -> None:
    """Sauvegarder la courbe ROC d'un modèle."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(5, 4))
    RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_precision_recall_curve(
    model: object,
    X_test,
    y_test,
    title: str,
    path: str | Path,
) -> None:
    """Sauvegarder la courbe Precision-Recall d'un modèle."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(5, 4))
    PrecisionRecallDisplay.from_estimator(model, X_test, y_test, ax=ax)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
