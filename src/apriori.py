"""
Módulo de Regras de Associação (Algoritmo Apriori) implementado 100% from scratch.
Sem dependências de bibliotecas de Machine Learning externas (proibido mlxtend / sklearn).
"""

from itertools import combinations
import pandas as pd
import numpy as np


def apriori(df, min_support=0.05, use_colnames=True, max_len=None):
    """
    Mineração de Itemsets Frequentes pelo algoritmo Apriori puro.
    
    Parâmetros:
    -----------
    df : pd.DataFrame
        DataFrame binário/booleano onde colunas são itens e linhas são transações.
    min_support : float
        Suporte mínimo no intervalo (0, 1].
    use_colnames : bool
        Se True, usa os nomes das colunas como nomes dos itens.
    max_len : int ou None
        Tamanho máximo dos itemsets a gerar.
        
    Retorna:
    --------
    pd.DataFrame com colunas ['support', 'itemsets']
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("df deve ser um pandas DataFrame binário/booleano.")

    n_transactions = len(df)
    if n_transactions == 0:
        return pd.DataFrame(columns=['support', 'itemsets'])

    columns = list(df.columns) if use_colnames else list(range(df.shape[1]))
    X = (df.values > 0).astype(bool)

    # 1. Itemsets de tamanho 1
    col_supports = np.mean(X, axis=0)
    freq_items_1 = {frozenset([columns[i]]): float(col_supports[i])
                    for i in range(len(columns)) if col_supports[i] >= min_support}

    all_frequent_itemsets = dict(freq_items_1)
    current_frequent = set(freq_items_1.keys())

    k = 2
    while current_frequent:
        if max_len is not None and k > max_len:
            break

        # Gerar candidatos de tamanho k a partir de itemsets frequentes de tamanho k-1
        items_in_curr = set().union(*current_frequent) if current_frequent else set()
        candidates = set()
        for itemset in current_frequent:
            for item in items_in_curr:
                if item not in itemset:
                    c = itemset | frozenset([item])
                    if len(c) == k:
                        # Poda Apriori: todos os subconjuntos de tamanho k-1 devem ser frequentes
                        all_subsets_freq = all(
                            frozenset(sub) in current_frequent
                            for sub in combinations(c, k - 1)
                        )
                        if all_subsets_freq:
                            candidates.add(c)

        if not candidates:
            break

        # Contagem de suporte dos candidatos
        next_frequent = {}
        col_to_idx = {col: idx for idx, col in enumerate(columns)}

        for cand in candidates:
            indices = [col_to_idx[item] for item in cand]
            # Uma transação contém o candidato se todas as colunas correspondentes forem True
            mask = np.all(X[:, indices], axis=1)
            supp = float(np.mean(mask))
            if supp >= min_support:
                next_frequent[cand] = supp

        all_frequent_itemsets.update(next_frequent)
        current_frequent = set(next_frequent.keys())
        k += 1

    if not all_frequent_itemsets:
        return pd.DataFrame(columns=['support', 'itemsets'])

    df_res = pd.DataFrame([
        {'support': supp, 'itemsets': itemset}
        for itemset, supp in all_frequent_itemsets.items()
    ])
    df_res = df_res.sort_values(by='support', ascending=False).reset_index(drop=True)
    return df_res


def association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5):
    """
    Gera regras de associação a partir de itemsets frequentes.
    
    Parâmetros:
    -----------
    frequent_itemsets : pd.DataFrame
        DataFrame retornado por apriori contendo ['support', 'itemsets'].
    metric : str
        Métrica de corte: 'confidence' ou 'lift'.
    min_threshold : float
        Limiar mínimo para a métrica selecionada.
        
    Retorna:
    --------
    pd.DataFrame com colunas detalhadas das regras de associação.
    """
    if frequent_itemsets.empty:
        return pd.DataFrame(columns=[
            'antecedents', 'consequents', 'antecedent_support',
            'consequent_support', 'support', 'confidence', 'lift',
            'leverage', 'conviction'
        ])

    supp_dict = {row['itemsets']: row['support'] for _, row in frequent_itemsets.iterrows()}
    rules = []

    for itemset, supp_both in supp_dict.items():
        if len(itemset) < 2:
            continue

        # Para cada partição não vazia A -> B
        for r in range(1, len(itemset)):
            for ante_tuple in combinations(itemset, r):
                ante = frozenset(ante_tuple)
                conseq = itemset - ante

                supp_ante = supp_dict.get(ante)
                supp_conseq = supp_dict.get(conseq)

                if supp_ante is None or supp_conseq is None:
                    continue

                conf = supp_both / supp_ante if supp_ante > 0 else 0.0
                lift = conf / supp_conseq if supp_conseq > 0 else 0.0
                leverage = supp_both - (supp_ante * supp_conseq)

                if conf < 1.0:
                    conviction = (1.0 - supp_conseq) / (1.0 - conf) if (1.0 - conf) > 0 else np.inf
                else:
                    conviction = np.inf

                rule = {
                    'antecedents': ante,
                    'consequents': conseq,
                    'antecedent_support': supp_ante,
                    'consequent_support': supp_conseq,
                    'support': supp_both,
                    'confidence': conf,
                    'lift': lift,
                    'leverage': leverage,
                    'conviction': conviction
                }

                if metric == 'confidence' and conf >= min_threshold:
                    rules.append(rule)
                elif metric == 'lift' and lift >= min_threshold:
                    rules.append(rule)

    if not rules:
        return pd.DataFrame(columns=[
            'antecedents', 'consequents', 'antecedent_support',
            'consequent_support', 'support', 'confidence', 'lift',
            'leverage', 'conviction'
        ])

    df_rules = pd.DataFrame(rules)
    df_rules = df_rules.sort_values(by=['confidence', 'lift'], ascending=[False, False]).reset_index(drop=True)
    return df_rules
