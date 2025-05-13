import numpy as np
import pytest
from data.models import AlmgrenChrissModel, SlippageModel, MakerTakerModel, FeeModel

def test_almgren_chriss_model():
    model = AlmgrenChrissModel(volatility=0.01, daily_volume=1000000)
    impact = model.estimate_impact(1000)
    assert impact > 0

def test_slippage_model():
    X = np.array([[1], [2], [3]])
    y = np.array([0.1, 0.2, 0.3])
    model = SlippageModel()
    model.fit(X, y)
    pred = model.predict([[2]])
    assert np.isclose(pred[0], 0.2, atol=0.05)

def test_maker_taker_model():
    X = np.array([[0], [1], [2], [3]])
    y = np.array([0, 1, 1, 0])
    model = MakerTakerModel()
    model.fit(X, y)
    proba = model.predict_proba([[1]])
    assert proba.shape == (1, 2)

def test_fee_model():
    fee = FeeModel.get_fee("Tier 1", "maker", 1000)
    assert np.isclose(fee, 0.8, atol=0.01) 