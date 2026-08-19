from collections import OrderedDict
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

def classification_models(random_state=42):
    m=OrderedDict()
    m["Logistic Regression"]=LogisticRegression(max_iter=3000,class_weight="balanced",random_state=random_state)
    m["Random Forest"]=RandomForestClassifier(n_estimators=400,class_weight="balanced",random_state=random_state,n_jobs=-1)
    m["SVM-RBF"]=SVC(kernel="rbf",probability=True,class_weight="balanced",random_state=random_state)
    try:
        from xgboost import XGBClassifier
        m["XGBoost"]=XGBClassifier(n_estimators=350,learning_rate=.05,max_depth=5,subsample=.9,colsample_bytree=.9,eval_metric="logloss",random_state=random_state,n_jobs=-1)
    except Exception: pass
    return m
