from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def numeric_preprocessor(features, scale=True):
    steps=[("imputer",SimpleImputer(strategy="median"))]
    if scale: steps.append(("scaler",StandardScaler()))
    return ColumnTransformer([("num",Pipeline(steps),list(features))], remainder="drop", verbose_feature_names_out=False)

def make_model_pipeline(model, numeric_features, scale=True):
    return Pipeline([("preprocessor",numeric_preprocessor(numeric_features,scale)),("model",model)])
