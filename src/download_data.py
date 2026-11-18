"""
Script de download e preparação dos datasets (FASE 0).
Descarrega os datasets públicos do Kaggle para 'data/' e prepara amostra estratificada.
"""

import os
import shutil
import numpy as np
import pandas as pd
import kagglehub

DATASET_SOURCES = {
    "housing": {
        "name": "California Housing Prices",
        "kaggle_slug": "camnugent/california-housing-prices",
        "url": "https://www.kaggle.com/datasets/camnugent/california-housing-prices",
        "description": "Dados censitários de habitação na Califórnia (1990) com variáveis demográficas, geográficas e valor mediano do imóvel.",
        "target_col": "median_house_value"
    },
    "home_credit": {
        "name": "Home Credit / Credit Risk Dataset",
        "kaggle_slug": "laotse/credit-risk-dataset",
        "url": "https://www.kaggle.com/datasets/laotse/credit-risk-dataset",
        "description": "Dados de crédito e histórico financeiro para previsão de risco de incumprimento (loan default risk).",
        "target_col": "loan_status"
    }
}


def download_and_prepare():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data")
    os.makedirs(data_dir, exist_ok=True)

    print("=== FASE 0: Download dos Datasets ===")

    # 1. Housing Dataset
    print(f"\n[1/2] Baixando Housing: {DATASET_SOURCES['housing']['kaggle_slug']}...")
    housing_cache_dir = kagglehub.dataset_download(DATASET_SOURCES['housing']['kaggle_slug'])
    print(f"Diretório baixado: {housing_cache_dir}")

    # Encontrar CSV
    housing_csv_src = None
    for root, _, files in os.walk(housing_cache_dir):
        for f in files:
            if f.lower().endswith(".csv"):
                housing_csv_src = os.path.join(root, f)
                break
        if housing_csv_src:
            break

    if not housing_csv_src:
        raise FileNotFoundError("Arquivo CSV de Housing não encontrado.")

    housing_dest = os.path.join(data_dir, "housing.csv")
    shutil.copyfile(housing_csv_src, housing_dest)
    df_housing = pd.read_csv(housing_dest)
    print(f"-> Salvo em {housing_dest} com {df_housing.shape[0]} linhas e {df_housing.shape[1]} colunas.")

    # 2. Home Credit Dataset
    print(f"\n[2/2] Baixando Home Credit / Credit Risk: {DATASET_SOURCES['home_credit']['kaggle_slug']}...")
    credit_cache_dir = kagglehub.dataset_download(DATASET_SOURCES['home_credit']['kaggle_slug'])
    print(f"Diretório baixado: {credit_cache_dir}")

    credit_csv_src = None
    for root, _, files in os.walk(credit_cache_dir):
        for f in files:
            if f.lower().endswith(".csv"):
                credit_csv_src = os.path.join(root, f)
                break
        if credit_csv_src:
            break

    if not credit_csv_src:
        raise FileNotFoundError("Arquivo CSV de Home Credit não encontrado.")

    credit_dest = os.path.join(data_dir, "home_credit.csv")
    shutil.copyfile(credit_csv_src, credit_dest)
    df_credit = pd.read_csv(credit_dest)
    print(f"-> Salvo em {credit_dest} com {df_credit.shape[0]} linhas e {df_credit.shape[1]} colunas.")

    # 3. Gerar amostra estratificada reprodutível se > 15.000 linhas
    if len(df_credit) > 15000:
        sample_size = 5000
        target = DATASET_SOURCES['home_credit']['target_col']
        print(f"\nDataset Home Credit possui {len(df_credit)} linhas (> 15.000).")
        print(f"Gerando amostra estratificada de {sample_size} linhas com random_state=42...")

        # Amostragem estratificada proporcional usando pandas
        frac = sample_size / len(df_credit)
        sample_parts = []
        for _, group in df_credit.groupby(target):
            sample_parts.append(group.sample(frac=frac, random_state=42))
        sample_df = pd.concat(sample_parts).sample(frac=1.0, random_state=42).reset_index(drop=True)

        sample_dest = os.path.join(data_dir, "home_credit_sample.csv")
        sample_df.to_csv(sample_dest, index=False)
        print(f"-> Amostra estratificada salva em {sample_dest} ({len(sample_df)} linhas).")
        print("Distribuição do alvo na amostra:")
        print(sample_df[target].value_counts(normalize=True))

    print("\nURLs de Origem Registadas:")
    for k, v in DATASET_SOURCES.items():
        print(f" - {v['name']}: {v['url']}")

    print("\nFASE 0 Concluída com Sucesso!")


if __name__ == "__main__":
    download_and_prepare()
