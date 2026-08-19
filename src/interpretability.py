import pandas as pd
from sklearn.inspection import permutation_importance

def permutation_importance_table(model,X,y,n_repeats=20,random_state=42):
    r=permutation_importance(model,X,y,n_repeats=n_repeats,random_state=random_state,scoring="f1",n_jobs=-1)
    return pd.DataFrame({"feature":X.columns,"importance_mean":r.importances_mean,"importance_std":r.importances_std}).sort_values("importance_mean",ascending=False).reset_index(drop=True)

def try_shap_tree(model_pipeline,X_sample):
    import shap
    pre=model_pipeline.named_steps["preprocessor"]; model=model_pipeline.named_steps["model"]
    Xt=pre.transform(X_sample); names=pre.get_feature_names_out(); explainer=shap.Explainer(model,Xt,feature_names=names)
    return explainer, explainer(Xt)
