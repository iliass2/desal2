"""Calcul des indices chimiques du projet."""
import numpy as np
import pandas as pd
_EQ = {"Cl": 35.453, "SO4": 48.03, "HCO3": 61.0168}

def mgL_to_meqL(values, ion: str):
    if ion not in _EQ: raise ValueError(f"Ion non pris en charge: {ion}")
    return pd.to_numeric(values, errors="coerce") / _EQ[ion]

def larson_index(cl_mgL, so4_mgL, hco3_mgL):
    cl = mgL_to_meqL(cl_mgL, "Cl")
    so4 = mgL_to_meqL(so4_mgL, "SO4")
    hco3 = mgL_to_meqL(hco3_mgL, "HCO3")
    if isinstance(hco3, pd.Series): hco3 = hco3.replace(0, np.nan)
    return (cl + 2 * so4) / hco3

def larson_class(index):
    return pd.cut(index, [-np.inf,.2,.4,.5,1,np.inf], labels=["Aucune","Faible","Légère","Moyenne","Nette"], right=False)

def larson_binary(index):
    s = pd.Series(index)
    out = (s >= 1).astype("Int64"); out[s.isna()] = pd.NA
    return out

def langelier_saturation_index(ph, tds_mgL, calcium_mgL, alkalinity_mgL_as_caco3, temperature_c):
    """Approximation LSI. Température obligatoire : aucune valeur par défaut n'est inventée."""
    ph = pd.to_numeric(ph, errors="coerce"); tds = pd.to_numeric(tds_mgL, errors="coerce")
    ca = pd.to_numeric(calcium_mgL, errors="coerce"); alk = pd.to_numeric(alkalinity_mgL_as_caco3, errors="coerce")
    temp = pd.to_numeric(temperature_c, errors="coerce")
    ca_hard = ca * (100.0869 / 40.078)
    A=(np.log10(tds)-1)/10; B=-13.12*np.log10(temp+273)+34.55
    C=np.log10(ca_hard)-0.4; D=np.log10(alk)
    pHs=(9.3+A+B)-(C+D)
    return ph-pHs

def add_larson_features(df):
    missing={"Cl","SO4","HCO3"}-set(df.columns)
    if missing: raise KeyError(f"Colonnes requises absentes: {sorted(missing)}")
    out=df.copy(); out["Larson_index"]=larson_index(out["Cl"],out["SO4"],out["HCO3"])
    out["Larson_class"]=larson_class(out["Larson_index"]); out["Larson_corrosive"]=larson_binary(out["Larson_index"])
    return out
