"""
Árvore de Decisão para Classificação (DecisionTreeClassifier)
Implementada 100% from scratch em Python e NumPy puro (vetorizada).
Sem dependências de bibliotecas de Machine Learning externas (proibido sklearn).
"""

import numpy as np


class Node:
    """
    Nó de uma Árvore de Decisão.
    """
    def __init__(self, is_leaf=False, value=None, probabilities=None,
                 feature_idx=None, threshold=None, left=None, right=None,
                 impurity=0.0, n_samples=0):
        self.is_leaf = is_leaf
        self.value = value                     # Classe predita (moda)
        self.probabilities = probabilities     # Distribuição de probabilidades de cada classe
        self.feature_idx = feature_idx         # Índice do atributo de partição
        self.threshold = threshold             # Limiar de corte
        self.left = left                       # Filho à esquerda (X[:, feature_idx] <= threshold)
        self.right = right                     # Filho à direita (X[:, feature_idx] > threshold)
        self.impurity = impurity               # Impureza no nó
        self.n_samples = n_samples             # Número de amostras no nó


class DecisionTreeClassifier:
    """
    Classificador por Árvore de Decisão construído do zero.
    
    Parâmetros:
    -----------
    criterion : str, 'gini' ou 'entropy' (default='gini')
    max_depth : int ou None (default=None)
    min_samples_split : int (default=2)
    min_samples_leaf : int (default=1)
    random_state : int ou None (default=None)
    """
    def __init__(self, criterion='gini', max_depth=None,
                 min_samples_split=2, min_samples_leaf=1, random_state=None):
        if criterion not in ('gini', 'entropy'):
            raise ValueError(f"Critério desconhecido: '{criterion}'. Use 'gini' ou 'entropy'.")
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = max(2, min_samples_split)
        self.min_samples_leaf = max(1, min_samples_leaf)
        self.random_state = random_state

        self.root = None
        self.classes_ = None
        self.n_classes_ = None
        self.n_features_ = None
        self.feature_importances_ = None

    def _impurity(self, counts, total):
        """
        Calcula a impureza (Gini ou Entropia) dado o array de contagens de classe.
        counts: array 1D ou 2D (vetorizado).
        total: escalar ou array de somas.
        """
        counts = np.asarray(counts, dtype=float)
        total = np.asarray(total, dtype=float)

        # Divisão segura evitando divisão por zero
        p = np.divide(counts, total, out=np.zeros_like(counts, dtype=float), where=(total > 0))

        if self.criterion == 'gini':
            if p.ndim == 1:
                return float(1.0 - np.sum(p ** 2))
            else:
                return 1.0 - np.sum(p ** 2, axis=-1)
        else:
            eps = 1e-12
            if p.ndim == 1:
                p_safe = p[p > eps]
                return float(-np.sum(p_safe * np.log2(p_safe))) if len(p_safe) > 0 else 0.0
            else:
                p_safe = np.where(p > eps, p, 1.0)
                entropy = -np.sum(np.where(p > eps, p * np.log2(p_safe), 0.0), axis=-1)
                return entropy

    def _best_split(self, X, y_one_hot):
        """
        Busca vetorizada da melhor divisão considerando todas as features e thresholds válidos.
        """
        n_samples, n_features = X.shape
        parent_counts = np.sum(y_one_hot, axis=0)
        parent_impurity = self._impurity(parent_counts, n_samples)

        best_gain = -1.0
        best_feat = None
        best_thresh = None

        for feat in range(n_features):
            x_col = X[:, feat]
            sort_idx = np.argsort(x_col)
            x_sorted = x_col[sort_idx]
            y_sorted = y_one_hot[sort_idx]

            # Detecta posições onde o valor consecutivo da feature muda
            split_mask = x_sorted[:-1] != x_sorted[1:]
            if not np.any(split_mask):
                continue

            # Contagens cumulativas vetorizadas à esquerda
            left_counts = np.cumsum(y_sorted, axis=0)[:-1]
            right_counts = parent_counts - left_counts

            n_left = np.arange(1, n_samples)
            n_right = n_samples - n_left

            # Filtrar respeito ao min_samples_leaf e valores distintos
            valid = split_mask & (n_left >= self.min_samples_leaf) & (n_right >= self.min_samples_leaf)
            if not np.any(valid):
                continue

            # Calcula impurezas ponderadas vetorizadamente para os splits válidos
            valid_indices = np.where(valid)[0]
            l_counts_v = left_counts[valid_indices]
            r_counts_v = right_counts[valid_indices]
            nl_v = n_left[valid_indices, None]
            nr_v = n_right[valid_indices, None]

            # Impurezas
            imp_left = self._impurity(l_counts_v, nl_v)
            imp_right = self._impurity(r_counts_v, nr_v)

            # Impureza ponderada
            weighted_imp = (nl_v.squeeze(-1) / n_samples) * imp_left + (nr_v.squeeze(-1) / n_samples) * imp_right
            gain = parent_impurity - weighted_imp

            max_idx_local = np.argmax(gain)
            max_gain = gain[max_idx_local]

            if max_gain > best_gain:
                best_gain = max_gain
                best_feat = feat
                split_point_idx = valid_indices[max_idx_local]
                best_thresh = (x_sorted[split_point_idx] + x_sorted[split_point_idx + 1]) / 2.0

        return best_feat, best_thresh, best_gain

    def _build_tree(self, X, y_int, y_one_hot, depth=0):
        n_samples = len(y_int)
        counts = np.bincount(y_int, minlength=self.n_classes_)
        majority_class = self.classes_[np.argmax(counts)]
        probs = counts / n_samples
        curr_impurity = self._impurity(counts, n_samples)

        # Condições de parada
        if (self.max_depth is not None and depth >= self.max_depth) or \
           n_samples < self.min_samples_split or \
           curr_impurity <= 1e-7 or \
           np.all(counts == 0):
            return Node(is_leaf=True, value=majority_class, probabilities=probs,
                        impurity=curr_impurity, n_samples=n_samples)

        # Buscar melhor split
        best_feat, best_thresh, best_gain = self._best_split(X, y_one_hot)

        if best_gain <= 1e-7 or best_feat is None:
            return Node(is_leaf=True, value=majority_class, probabilities=probs,
                        impurity=curr_impurity, n_samples=n_samples)

        # Acumular ganho na importância das features
        self._raw_importances[best_feat] += best_gain * (n_samples / self._total_samples)

        left_mask = X[:, best_feat] <= best_thresh
        right_mask = ~left_mask

        # Caso degenerado (não deve ocorrer com o filtro min_samples_leaf)
        if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
            return Node(is_leaf=True, value=majority_class, probabilities=probs,
                        impurity=curr_impurity, n_samples=n_samples)

        left_child = self._build_tree(X[left_mask], y_int[left_mask], y_one_hot[left_mask], depth + 1)
        right_child = self._build_tree(X[right_mask], y_int[right_mask], y_one_hot[right_mask], depth + 1)

        return Node(is_leaf=False, value=majority_class, probabilities=probs,
                    feature_idx=best_feat, threshold=best_thresh,
                    left=left_child, right=right_child,
                    impurity=curr_impurity, n_samples=n_samples)

    def fit(self, X, y):
        """
        Treina o classificador de árvore de decisão.
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)

        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y)

        self.classes_ = np.unique(y_arr)
        self.n_classes_ = len(self.classes_)
        self.n_features_ = X_arr.shape[1]
        self._total_samples = len(y_arr)
        self._raw_importances = np.zeros(self.n_features_, dtype=float)

        # Mapeamento de y para inteiros 0..C-1
        label_to_idx = {cls: idx for idx, cls in enumerate(self.classes_)}
        y_int = np.array([label_to_idx[val] for val in y_arr], dtype=int)

        # Matriz One-Hot para operações vetorizadas de soma
        y_one_hot = np.zeros((self._total_samples, self.n_classes_), dtype=float)
        y_one_hot[np.arange(self._total_samples), y_int] = 1.0

        self.root = self._build_tree(X_arr, y_int, y_one_hot, depth=0)

        # Normalizar importâncias das features
        sum_imp = np.sum(self._raw_importances)
        if sum_imp > 0:
            self.feature_importances_ = self._raw_importances / sum_imp
        else:
            self.feature_importances_ = np.zeros(self.n_features_, dtype=float)

        return self

    def _predict_row_proba(self, node, row):
        if node.is_leaf:
            return node.probabilities
        if row[node.feature_idx] <= node.threshold:
            return self._predict_row_proba(node.left, row)
        else:
            return self._predict_row_proba(node.right, row)

    def predict_proba(self, X):
        """
        Retorna as probabilidades preditas para cada classe.
        """
        X_arr = np.asarray(X, dtype=float)
        return np.array([self._predict_row_proba(self.root, row) for row in X_arr])

    def predict(self, X):
        """
        Retorna a classe predita de maior probabilidade.
        """
        probas = self.predict_proba(X)
        best_indices = np.argmax(probas, axis=1)
        return self.classes_[best_indices]

    def score(self, X, y):
        """
        Retorna a acurácia média nos dados de teste fornecidos.
        """
        preds = self.predict(X)
        return float(np.mean(preds == np.asarray(y)))

    def export_text(self, feature_names=None, max_depth=5):
        """
        Gera uma representação textual da estrutura da árvore.
        """
        if feature_names is None:
            feature_names = [f"X[{i}]" for i in range(self.n_features_)]

        lines = []
        def _recurse(node, depth=0, prefix=""):
            if depth > max_depth:
                lines.append(f"{prefix}... (árvore truncada para visualização)")
                return
            indent = "  " * depth
            if node.is_leaf:
                lines.append(f"{indent}|--- Classe: {node.value} (amostras: {node.n_samples}, prob: {np.round(node.probabilities, 3)})")
            else:
                fname = feature_names[node.feature_idx]
                lines.append(f"{indent}|--- Se {fname} <= {node.threshold:.4f}:")
                _recurse(node.left, depth + 1)
                lines.append(f"{indent}|--- Senão ({fname} > {node.threshold:.4f}):")
                _recurse(node.right, depth + 1)

        _recurse(self.root)
        return "\n".join(lines)
