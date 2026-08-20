import math

import numpy as np
import pandas as pd
import pytest

from src.indices_chimiques import (
    larson_binary,
    larson_class,
    larson_index,
    mgL_to_meqL,
)


def test_mgL_to_meqL_chloride_one_equivalent():
    result = mgL_to_meqL(35.453, "Cl")
    assert result == pytest.approx(1.0)


def test_mgL_to_meqL_rejects_unknown_ion():
    with pytest.raises(ValueError):
        mgL_to_meqL(10.0, "Na")


def test_larson_index_known_values():
    # 1 meq/L Cl + 2 * 1 meq/L SO4, divisé par 1 meq/L HCO3 = 3
    result = larson_index(35.453, 48.03, 61.0168)
    assert result == pytest.approx(3.0)


def test_larson_zero_bicarbonate_returns_nan():
    result = larson_index(35.453, 48.03, 0.0)
    assert math.isnan(result)


def test_larson_classes_at_representative_values():
    values = pd.Series([0.10, 0.30, 0.45, 0.75, 1.20])
    classes = larson_class(values).astype(str).tolist()
    assert classes == ["Aucune", "Faible", "Légère", "Moyenne", "Nette"]


def test_larson_binary_threshold():
    values = pd.Series([0.99, 1.00, 2.00, np.nan])
    result = larson_binary(values)
    assert result.iloc[0] == 0
    assert result.iloc[1] == 1
    assert result.iloc[2] == 1
    assert pd.isna(result.iloc[3])
