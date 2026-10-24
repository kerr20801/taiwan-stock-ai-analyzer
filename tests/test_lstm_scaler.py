"""create_sequences：訓練時 scaler 只能看到訓練段；評估時不得重新 fit。TensorFlow 以 stub 取代，不需安裝。"""
import sys
import types
from unittest.mock import MagicMock

import numpy as np
import pandas as pd

for name in ("tensorflow", "tensorflow.keras", "tensorflow.keras.models",
             "tensorflow.keras.layers", "tensorflow.keras.callbacks"):
    sys.modules.setdefault(name, MagicMock())

import lstm_stock_predictor as m  # noqa: E402


def _predictor(seq_len=5):
    p = object.__new__(m.StockLSTMPredictor)
    p.sequence_length = seq_len
    p.scaler = m.MinMaxScaler(feature_range=(0, 1))
    return p


def _df(n=100):
    close = np.arange(n, dtype=float)
    close[-1] = 10_000.0  # 驗證段的極端值：不得進入 scaler
    return pd.DataFrame({"close": close, "volume": np.ones(n)})


def test_train_fit_excludes_validation_rows():
    p, df = _predictor(), _df()
    p.create_sequences(df, validation_split=0.2)
    n_fit = 5 + int((100 - 5) * 0.8)
    assert p.scaler.data_max_[0] == df["close"].iloc[n_fit - 1]
    assert p.scaler.data_max_[0] < 10_000.0


def test_no_validation_split_does_not_refit():
    p, df = _predictor(), _df()
    p.create_sequences(df, validation_split=0.2)
    before = p.scaler.data_max_.copy()
    p.create_sequences(df)  # 評估／預測路徑
    assert (p.scaler.data_max_ == before).all()


def test_last_train_target_row_is_in_fit_window():
    p, df = _predictor(), _df()
    _, y = p.create_sequences(df, validation_split=0.2)
    split_idx = int(len(y) * 0.8)
    assert y[split_idx - 1] <= 1.0  # 最後一筆訓練目標在 fit 範圍內
    assert y[split_idx:].max() > 1.0  # 驗證段超出 scaler 範圍 = 沒有洩漏
