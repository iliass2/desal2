"""Calcul des indices chimiques utilisés dans le projet."""

import numpy as np
import pandas as pd

# Masses équivalentes en mg/meq.
# Cl- : M / |z| = 35.453 / 1
# SO4^2- : M / |z| = 96.06 / 2 = 48.03
# HCO3- : M / |z| = 61.0168 / 1
_EQUIVALENT_MASS_MG_PER_MEQ = {
    "Cl": 35.453,
    "SO4": 48.03,
    "HCO3": 61.0168,
}


def mgL_to_meqL(values, ion: str):
    """Convertir une concentration de mg/L vers meq/L pour un ion supporté."""
    if ion not in _EQUIVALENT_MASS_MG_PER_MEQ:
        raise ValueError(f"Ion non pris en charge : {ion}")

    numeric = pd.to_numeric(values, errors="coerce")
    return numeric / _EQUIVALENT_MASS_MG_PER_MEQ[ion]


def larson_index(cl_mgL, so4_mgL, hco3_mgL):
    """Calculer l'indice de Larson après conversion des ions en meq/L.

    Formule utilisée dans le projet :
        IC = (Cl + 2 * SO4) / HCO3

    Les trois concentrations sont converties en meq/L avant le calcul.
    Un bicarbonate nul est remplacé par NaN pour éviter une division par zéro.
    """
    chloride = mgL_to_meqL(cl_mgL, "Cl")
    sulfate = mgL_to_meqL(so4_mgL, "SO4")
    bicarbonate = mgL_to_meqL(hco3_mgL, "HCO3")

    if isinstance(bicarbonate, pd.Series):
        bicarbonate = bicarbonate.replace(0, np.nan)
    elif np.isscalar(bicarbonate) and bicarbonate == 0:
        bicarbonate = np.nan

    return (chloride + 2 * sulfate) / bicarbonate


def larson_class(index):
    """Classer l'indice de Larson selon les seuils retenus dans le projet."""
    return pd.cut(
        index,
        bins=[-np.inf, 0.2, 0.4, 0.5, 1.0, np.inf],
        labels=["Aucune", "Faible", "Légère", "Moyenne", "Nette"],
        right=False,
    )


def larson_binary(index):
    """Créer une cible binaire : 1 si IC >= 1, sinon 0."""
    series = pd.Series(index, dtype="Float64")
    output = (series >= 1.0).astype("Int64")
    output[series.isna()] = pd.NA
    return output


def langelier_saturation_index(
    ph,
    tds_mgL,
    calcium_mgL,
    alkalinity_mgL_as_caco3,
    temperature_c,
):
    """Calculer une approximation du LSI avec température obligatoire.

    Aucune température par défaut n'est fournie volontairement : le projet ne
    doit pas inventer une valeur absente du jeu de données.
    """
    ph = pd.to_numeric(ph, errors="coerce")
    tds = pd.to_numeric(tds_mgL, errors="coerce")
    calcium = pd.to_numeric(calcium_mgL, errors="coerce")
    alkalinity = pd.to_numeric(alkalinity_mgL_as_caco3, errors="coerce")
    temperature = pd.to_numeric(temperature_c, errors="coerce")

    calcium_hardness = calcium * (100.0869 / 40.078)

    a_term = (np.log10(tds) - 1) / 10
    b_term = -13.12 * np.log10(temperature + 273) + 34.55
    c_term = np.log10(calcium_hardness) - 0.4
    d_term = np.log10(alkalinity)

    ph_saturation = (9.3 + a_term + b_term) - (c_term + d_term)
    return ph - ph_saturation


def add_larson_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Ajouter indice, classe qualitative et cible binaire de Larson."""
    required = {"Cl", "SO4", "HCO3"}
    missing = required - set(dataframe.columns)
    if missing:
        raise KeyError(f"Colonnes requises absentes : {sorted(missing)}")

    output = dataframe.copy()
    output["Larson_index"] = larson_index(
        output["Cl"],
        output["SO4"],
        output["HCO3"],
    )
    output["Larson_class"] = larson_class(output["Larson_index"])
    output["Larson_corrosive"] = larson_binary(output["Larson_index"])
    return output
