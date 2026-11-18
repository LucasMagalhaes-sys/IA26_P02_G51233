"""
Módulo de métricas de avaliação implementado em NumPy puro.
Sem dependências de bibliotecas de Machine Learning externas (proibido sklearn).
"""

import numpy as np


# ==========================================
# MÉTRICAS DE CLASSIFICAÇÃO
# ==========================================

def confusion_matrix(y_true, y_pred, labels=None):
    """
    Calcula a matriz de confusão para problemas de classificação.
    Retorna array 2D onde a linha i representa a classe real i
    e a coluna j representa a classe predita j.
    """
    y_t = np.asarray(y_true)
    y_p = np.asarray(y_pred)

    if labels is None:
        labels = np.unique(np.concatenate([y_t, y_p]))
    labels = np.asarray(labels)

    n_labels = len(labels)
    label_to_idx = {label: idx for idx, label in enumerate(labels)}

    cm = np.zeros((n_labels, n_labels), dtype=int)
    for t, p in zip(y_t, y_p):
        if t in label_to_idx and p in label_to_idx:
            cm[label_to_idx[t], label_to_idx[p]] += 1

    return cm


def accuracy_score(y_true, y_pred):
    """
    Calcula a acurácia de classificação: (Acertos / Total).
    """
    y_t = np.asarray(y_true)
    y_p = np.asarray(y_pred)
    if len(y_t) == 0:
        return 0.0
    return float(np.mean(y_t == y_p))


def precision_score(y_true, y_pred, pos_label=1, average='binary', zero_division=0.0):
    """
    Calcula a precisão: TP / (TP + FP).
    Suporta 'binary' e 'macro'.
    """
    y_t = np.asarray(y_true)
    y_p = np.asarray(y_pred)

    if average == 'binary':
        tp = np.sum((y_t == pos_label) & (y_p == pos_label))
        fp = np.sum((y_t != pos_label) & (y_p == pos_label))
        denom = tp + fp
        return float(tp / denom) if denom > 0 else float(zero_division)

    elif average in ('macro', 'weighted'):
        classes = np.unique(np.concatenate([y_t, y_p]))
        precisions = []
        weights = []
        for c in classes:
            tp = np.sum((y_t == c) & (y_p == c))
            fp = np.sum((y_t != c) & (y_p == c))
            denom = tp + fp
            p = tp / denom if denom > 0 else zero_division
            precisions.append(p)
            weights.append(np.sum(y_t == c))

        if average == 'macro':
            return float(np.mean(precisions))
        else:
            w_sum = np.sum(weights)
            return float(np.sum(np.array(precisions) * np.array(weights)) / w_sum) if w_sum > 0 else 0.0

    raise ValueError(f"average='{average}' não suportado. Use 'binary' ou 'macro'.")


def recall_score(y_true, y_pred, pos_label=1, average='binary', zero_division=0.0):
    """
    Calcula a revocação / sensibilidade: TP / (TP + FN).
    Suporta 'binary' e 'macro'.
    """
    y_t = np.asarray(y_true)
    y_p = np.asarray(y_pred)

    if average == 'binary':
        tp = np.sum((y_t == pos_label) & (y_p == pos_label))
        fn = np.sum((y_t == pos_label) & (y_p != pos_label))
        denom = tp + fn
        return float(tp / denom) if denom > 0 else float(zero_division)

    elif average in ('macro', 'weighted'):
        classes = np.unique(np.concatenate([y_t, y_p]))
        recalls = []
        weights = []
        for c in classes:
            tp = np.sum((y_t == c) & (y_p == c))
            fn = np.sum((y_t == c) & (y_p != c))
            denom = tp + fn
            r = tp / denom if denom > 0 else zero_division
            recalls.append(r)
            weights.append(np.sum(y_t == c))

        if average == 'macro':
            return float(np.mean(recalls))
        else:
            w_sum = np.sum(weights)
            return float(np.sum(np.array(recalls) * np.array(weights)) / w_sum) if w_sum > 0 else 0.0

    raise ValueError(f"average='{average}' não suportado. Use 'binary' ou 'macro'.")


def f1_score(y_true, y_pred, pos_label=1, average='binary', zero_division=0.0):
    """
    Calcula o F1-Score: média harmônica entre precisão e revocação.
    """
    p = precision_score(y_true, y_pred, pos_label=pos_label, average=average, zero_division=zero_division)
    r = recall_score(y_true, y_pred, pos_label=pos_label, average=average, zero_division=zero_division)
    denom = p + r
    return float(2 * (p * r) / denom) if denom > 0 else float(zero_division)


def classification_report(y_true, y_pred, digits=4):
    """
    Gera um relatório descritivo textual das métricas de classificação.
    """
    y_t = np.asarray(y_true)
    y_p = np.asarray(y_pred)
    classes = np.unique(np.concatenate([y_t, y_p]))

    header = f"{'Classe':>10} | {'Precisão':>{digits+6}} | {'Revocação':>{digits+6}} | {'F1-Score':>{digits+6}} | {'Suporte':>10}"
    separator = "-" * len(header)
    lines = [header, separator]

    total_samples = len(y_t)
    precisions, recalls, f1s, supports = [], [], [], []

    for c in classes:
        p = precision_score(y_t, y_p, pos_label=c, average='binary')
        r = recall_score(y_t, y_p, pos_label=c, average='binary')
        denom = p + r
        f = (2 * p * r / denom) if denom > 0 else 0.0
        s = int(np.sum(y_t == c))

        precisions.append(p)
        recalls.append(r)
        f1s.append(f)
        supports.append(s)

        lines.append(f"{str(c):>10} | {p:>{digits+6}.{digits}f} | {r:>{digits+6}.{digits}f} | {f:>{digits+6}.{digits}f} | {s:>10d}")

    lines.append(separator)
    acc = accuracy_score(y_t, y_p)
    lines.append(f"{'Acurácia':>10} | {' ':>{digits+6}} | {' ':>{digits+6}} | {acc:>{digits+6}.{digits}f} | {total_samples:>10d}")

    macro_p = np.mean(precisions)
    macro_r = np.mean(recalls)
    macro_f = np.mean(f1s)
    lines.append(f"{'Macro Avg':>10} | {macro_p:>{digits+6}.{digits}f} | {macro_r:>{digits+6}.{digits}f} | {macro_f:>{digits+6}.{digits}f} | {total_samples:>10d}")

    weights = np.array(supports) / total_samples
    w_p = np.sum(np.array(precisions) * weights)
    w_r = np.sum(np.array(recalls) * weights)
    w_f = np.sum(np.array(f1s) * weights)
    lines.append(f"{'Ponderada':>10} | {w_p:>{digits+6}.{digits}f} | {w_r:>{digits+6}.{digits}f} | {w_f:>{digits+6}.{digits}f} | {total_samples:>10d}")

    return "\n".join(lines)


# ==========================================
# MÉTRICAS DE CLUSTERING
# ==========================================

def wcss(X, labels, centroids=None):
    """
    Calcula WCSS (Within-Cluster Sum of Squares) / Inércia:
    Soma das distâncias euclidianas ao quadrado de cada ponto ao seu respectivo centroide.
    """
    X_arr = np.asarray(X, dtype=float)
    lbl = np.asarray(labels, dtype=int)
    unique_labels = np.unique(lbl)

    if centroids is None:
        centroids = np.zeros((len(unique_labels), X_arr.shape[1]))
        for idx, k in enumerate(unique_labels):
            mask = (lbl == k)
            if np.any(mask):
                centroids[idx] = np.mean(X_arr[mask], axis=0)

    total_wcss = 0.0
    for idx, k in enumerate(unique_labels):
        cluster_points = X_arr[lbl == k]
        if len(cluster_points) > 0:
            center = centroids[idx] if idx < len(centroids) else np.mean(cluster_points, axis=0)
            diffs = cluster_points - center
            total_wcss += np.sum(diffs ** 2)

    return float(total_wcss)


def silhouette_score(X, labels, sample_size=None, random_state=None):
    """
    Calcula o Silhouette Score médio em NumPy puro.
    Suporta sub-amostragem opcional (sample_size) para rápida convergência em grandes volumes.
    Retorna valor entre -1 e +1.
    """
    X_arr = np.asarray(X, dtype=float)
    lbl = np.asarray(labels)
    unique_labels = np.unique(lbl)

    if len(unique_labels) < 2 or len(unique_labels) >= len(X_arr):
        return 0.0

    if sample_size is not None and sample_size < len(X_arr):
        if random_state is not None:
            np.random.seed(random_state)
        idx = np.random.choice(len(X_arr), size=sample_size, replace=False)
        X_sample = X_arr[idx]
        lbl_sample = lbl[idx]
    else:
        X_sample = X_arr
        lbl_sample = lbl

    n_samples = len(X_sample)
    # Matriz de distâncias euclidianas do subconjunto
    # ||u - v|| = sqrt(||u||^2 + ||v||^2 - 2 u.v)
    dots = np.dot(X_sample, X_sample.T)
    sq_norms = np.diag(dots)
    dist_matrix_sq = np.maximum(sq_norms[:, None] + sq_norms[None, :] - 2 * dots, 0.0)
    dist_matrix = np.sqrt(dist_matrix_sq)

    silhouettes = np.zeros(n_samples)

    for i in range(n_samples):
        curr_label = lbl_sample[i]
        same_cluster_mask = (lbl_sample == curr_label)
        same_cluster_size = np.sum(same_cluster_mask)

        if same_cluster_size <= 1:
            silhouettes[i] = 0.0
            continue

        # a(i): distância média para outros pontos do mesmo cluster
        a_i = np.sum(dist_matrix[i, same_cluster_mask]) / (same_cluster_size - 1)

        # b(i): menor distância média para pontos de qualquer outro cluster
        b_i = np.inf
        for other_label in unique_labels:
            if other_label == curr_label:
                continue
            other_mask = (lbl_sample == other_label)
            if np.any(other_mask):
                dist_to_other = np.mean(dist_matrix[i, other_mask])
                if dist_to_other < b_i:
                    b_i = dist_to_other

        max_ab = max(a_i, b_i)
        silhouettes[i] = (b_i - a_i) / max_ab if max_ab > 0 else 0.0

    return float(np.mean(silhouettes))
