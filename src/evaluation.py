from pathlib import Path
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score,ConfusionMatrixDisplay,RocCurveDisplay
from sklearn.model_selection import StratifiedKFold,cross_validate

def classification_metrics(model,X_test,y_test):
    pred=model.predict(X_test); r={"accuracy":accuracy_score(y_test,pred),"precision":precision_score(y_test,pred,zero_division=0),"recall":recall_score(y_test,pred,zero_division=0),"f1":f1_score(y_test,pred,zero_division=0)}
    if hasattr(model,"predict_proba"): r["roc_auc"]=roc_auc_score(y_test,model.predict_proba(X_test)[:,1])
    elif hasattr(model,"decision_function"): r["roc_auc"]=roc_auc_score(y_test,model.decision_function(X_test))
    else: r["roc_auc"]=np.nan
    return r

def cross_validation_table(model,X,y,n_splits=5,random_state=42):
    cv=StratifiedKFold(n_splits=n_splits,shuffle=True,random_state=random_state)
    scoring=["accuracy","precision","recall","f1","roc_auc"]
    s=cross_validate(model,X,y,cv=cv,scoring=scoring,n_jobs=-1)
    return pd.DataFrame([{"metric":k,"mean":s[f"test_{k}"].mean(),"std":s[f"test_{k}"].std()} for k in scoring])

def save_confusion_matrix(model,X_test,y_test,title,path):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); fig,ax=plt.subplots(figsize=(5,4))
    ConfusionMatrixDisplay.from_estimator(model,X_test,y_test,ax=ax,cmap="Blues",colorbar=False); ax.set_title(title); fig.tight_layout(); fig.savefig(path,dpi=160,bbox_inches="tight"); plt.close(fig)

def save_roc_curve(model,X_test,y_test,title,path):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); fig,ax=plt.subplots(figsize=(5,4))
    RocCurveDisplay.from_estimator(model,X_test,y_test,ax=ax); ax.set_title(title); fig.tight_layout(); fig.savefig(path,dpi=160,bbox_inches="tight"); plt.close(fig)
