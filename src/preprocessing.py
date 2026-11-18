"""
Módulo de pré-processamento em NumPy e Pandas puro.
Sem dependências de bibliotecas de Machine Learning externas (proibido sklearn).
"""

import numpy as np
import pandas as pd


def train_test_split(X, y, test_size=0.2, random_state=None, shuffle=True, stratify=None):
    """
    Divide arrays ou DataFrames em conjuntos de treino e teste.
    Suporta amostragem aleatória e amostragem estratificada.
    
    Retorna: X_train, X_test, y_train, y_test
    """
    if random_state is not None:
        np.random.seed(random_state)

    # Conversão para arrays mantendo indexação se necessário
    is_df_x = isinstance(X, (pd.DataFrame, pd.Series))
    is_df_y = isinstance(y, (pd.DataFrame, pd.Series))

    n_samples = len(X)
    if len(y) != n_samples:
        raise ValueError(f"X e y têm comprimentos diferentes: {n_samples} vs {len(y)}")

    if stratify is not None:
        strat_arr = np.asarray(stratify)
        classes, class_counts = np.unique(strat_arr, return_counts=True)
        train_indices = []
        test_indices = []

        for cls in classes:
            cls_indices = np.where(strat_arr == cls)[0]
            if shuffle:
                np.random.shuffle(cls_indices)
            n_test_cls = int(np.round(len(cls_indices) * test_size))
            test_indices.extend(cls_indices[:n_test_cls])
            train_indices.extend(cls_indices[n_test_cls:])

        train_indices = np.array(train_indices)
        test_indices = np.array(test_indices)
        if shuffle:
            np.random.shuffle(train_indices)
            np.random.shuffle(test_indices)
    else:
        indices = np.arange(n_samples)
        if shuffle:
            np.random.shuffle(indices)
        n_test = int(np.round(n_samples * test_size))
        test_indices = indices[:n_test]
        train_indices = indices[n_test:]

    if is_df_x:
        X_train, X_test = X.iloc[train_indices], X.iloc[test_indices]
    else:
        X_arr = np.asarray(X)
        X_train, X_test = X_arr[train_indices], X_arr[test_indices]

    if is_df_y:
        y_train, y_test = y.iloc[train_indices], y.iloc[test_indices]
    else:
        y_arr = np.asarray(y)
        y_train, y_test = y_arr[train_indices], y_arr[test_indices]

    return X_train, X_test, y_train, y_test


class StandardScaler:
    """
    Padronizador de variáveis: z = (x - u) / s
    Calculado e transformado exclusivamente em NumPy puro.
    """
    def __init__(self, eps=1e-8):
        self.eps = eps
        self.mean_ = None
        self.scale_ = None

    def fit(self, X):
        X_arr = np.asarray(X, dtype=float)
        self.mean_ = np.nanmean(X_arr, axis=0)
        self.scale_ = np.nanstd(X_arr, axis=0)
        # Evita divisão por zero
        self.scale_[self.scale_ < self.eps] = 1.0
        return self

    def transform(self, X):
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("O StandardScaler ainda não foi ajustado (use fit antes de transform).")
        X_arr = np.asarray(X, dtype=float)
        return (X_arr - self.mean_) / self.scale_

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def inverse_transform(self, X):
        X_arr = np.asarray(X, dtype=float)
        return (X_arr * self.scale_) + self.mean_


class MinMaxScaler:
    """
    Normalizador Min-Max: x_norm = (x - min) / (max - min)
    Garante valores no intervalo [feature_range[0], feature_range[1]].
    """
    def __init__(self, feature_range=(0, 1), eps=1e-8):
        self.feature_range = feature_range
        self.eps = eps
        self.data_min_ = None
        self.data_max_ = None

    def fit(self, X):
        X_arr = np.asarray(X, dtype=float)
        self.data_min_ = np.nanmin(X_arr, axis=0)
        self.data_max_ = np.nanmax(X_arr, axis=0)
        return self

    def transform(self, X):
        if self.data_min_ is None:
            raise RuntimeError("MinMaxScaler não foi ajustado.")
        X_arr = np.asarray(X, dtype=float)
        rng = self.data_max_ - self.data_min_
        rng[rng < self.eps] = 1.0
        norm = (X_arr - self.data_min_) / rng
        return norm * (self.feature_range[1] - self.feature_range[0]) + self.feature_range[0]

    def fit_transform(self, X):
        return self.fit(X).transform(X)


class LabelEncoder:
    """
    Codificador ordinal de classes discretas/rótulos para inteiros [0, n_classes-1].
    """
    def __init__(self):
        self.classes_ = None
        self._mapping = {}
        self._inv_mapping = {}

    def fit(self, y):
        arr = np.asarray(y)
        self.classes_ = np.unique(arr)
        self._mapping = {cls: idx for idx, cls in enumerate(self.classes_)}
        self._inv_mapping = {idx: cls for idx, cls in enumerate(self.classes_)}
        return self

    def transform(self, y):
        arr = np.asarray(y)
        return np.array([self._mapping[val] for val in arr], dtype=int)

    def fit_transform(self, y):
        return self.fit(y).transform(y)

    def inverse_transform(self, y):
        arr = np.asarray(y, dtype=int)
        return np.array([self._inv_mapping[val] for val in arr])


def impute_missing(df, strategy='median', categorical_strategy='mode'):
    """
    Tratamento de valores em falta em DataFrames Pandas:
    - Colunas numéricas: imputação por mediana ou média.
    - Colunas categóricas: imputação pela moda (valor mais frequente).
    """
    df_clean = df.copy()
    for col in df_clean.columns:
        if pd.api.types.is_numeric_dtype(df_clean[col]):
            if df_clean[col].isnull().any():
                if strategy == 'median':
                    fill_val = df_clean[col].median()
                elif strategy == 'mean':
                    fill_val = df_clean[col].mean()
                else:
                    fill_val = 0
                df_clean[col] = df_clean[col].fillna(fill_val)
        else:
            if df_clean[col].isnull().any():
                if categorical_strategy == 'mode':
                    mode_val = df_clean[col].mode()[0]
                    df_clean[col] = df_clean[col].fillna(mode_val)
                else:
                    df_clean[col] = df_clean[col].fillna("Missing")
    return df_clean


def encode_categorical_features(df, categorical_cols=None):
    """
    One-Hot Encoding simples e reproduzível para colunas categóricas.
    """
    df_encoded = df.copy()
    if categorical_cols is None:
        categorical_cols = df_encoded.select_dtypes(include=['object', 'category', 'string']).columns.tolist()

    df_encoded = pd.get_dummies(df_encoded, columns=categorical_cols, drop_first=True, dtype=float)
    return df_encoded
