"""Définition des modèles de classification comparés dans le PFE."""

from collections import OrderedDict

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


def classification_models(random_state: int = 42) -> OrderedDict:
    """Retourner un ensemble ordonné de modèles incluant une baseline naïve."""
    models = OrderedDict()

    models["Dummy Classifier"] = DummyClassifier(strategy="most_frequent")
    models["Logistic Regression"] = LogisticRegression(
        max_iter=3000,
        class_weight="balanced",
        random_state=random_state,
    )
    models["Random Forest"] = RandomForestClassifier(
        n_estimators=400,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )
    models["SVM-RBF"] = SVC(
        kernel="rbf",
        probability=True,
        class_weight="balanced",
        random_state=random_state,
    )

    try:
        from xgboost import XGBClassifier

        models["XGBoost"] = XGBClassifier(
            n_estimators=350,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=random_state,
            n_jobs=-1,
        )
    except ImportError:
        # Le projet reste exécutable sans xgboost.
        pass

    return models


def needs_scaling(model_name: str) -> bool:
    """Indiquer si le modèle bénéficie d'une standardisation des prédicteurs."""
    return model_name in {"Logistic Regression", "SVM-RBF"}
