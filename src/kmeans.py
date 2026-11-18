"""
Algoritmo de Agrupamento K-Means implementado 100% from scratch em Python e NumPy puro.
Sem dependências de bibliotecas de Machine Learning externas (proibido sklearn).
"""

import numpy as np


class KMeans:
    """
    K-Means Clustering construído do zero com Python e NumPy.
    
    Parâmetros:
    -----------
    n_clusters : int, default=8
        Número de clusters desejado (K).
    init : str, 'k-means++' ou 'random' (default='k-means++')
        Estratégia de inicialização dos centroides.
    max_iter : int, default=300
        Número máximo de iterações do algoritmo de Lloyd.
    tol : float, default=1e-4
        Tolerância para convergência com base no deslocamento dos centroides.
    random_state : int ou None, default=None
        Semente para reprodutibilidade estrita.
    """
    def __init__(self, n_clusters=8, init='k-means++', max_iter=300, tol=1e-4, random_state=None):
        if n_clusters < 1:
            raise ValueError(f"n_clusters deve ser >= 1, recebido: {n_clusters}")
        if init not in ('k-means++', 'random'):
            raise ValueError(f"Estratégia init='{init}' desconhecida. Use 'k-means++' ou 'random'.")
        self.n_clusters = n_clusters
        self.init = init
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

        self.cluster_centers_ = None
        self.labels_ = None
        self.inertia_ = None
        self.inertia_history_ = []
        self.n_iter_ = 0

    def _init_centroids(self, X):
        n_samples, n_features = X.shape
        rng = np.random.RandomState(self.random_state)

        if self.init == 'random':
            indices = rng.choice(n_samples, size=self.n_clusters, replace=False)
            return X[indices].copy()

        # Estratégia K-Means++
        centroids = np.empty((self.n_clusters, n_features), dtype=float)
        # Primeiro centroide aleatório
        first_idx = rng.choice(n_samples)
        centroids[0] = X[first_idx]

        for k in range(1, self.n_clusters):
            # Distância ao quadrado de cada ponto ao centroide mais próximo
            # ||x - c||^2
            dists = np.min([np.sum((X - centroids[j]) ** 2, axis=1) for j in range(k)], axis=0)
            dists = np.maximum(dists, 0.0)
            total_dist = np.sum(dists)

            if total_dist == 0:
                # Se todos os pontos restantes coincidirem com centroides existentes
                probs = np.ones(n_samples) / n_samples
            else:
                probs = dists / total_dist

            next_idx = rng.choice(n_samples, p=probs)
            centroids[k] = X[next_idx]

        return centroids

    def _compute_distances(self, X, centroids):
        """
        Calcula as distâncias euclidianas ao quadrado de cada ponto a cada centroide.
        Forma: (n_samples, n_clusters)
        """
        # ||x - c||^2 = ||x||^2 + ||c||^2 - 2 * x . c^T
        x_sq = np.sum(X ** 2, axis=1, keepdims=True)
        c_sq = np.sum(centroids ** 2, axis=1, keepdims=True).T
        xc = np.dot(X, centroids.T)
        dist_sq = np.maximum(x_sq + c_sq - 2 * xc, 0.0)
        return dist_sq

    def fit(self, X):
        """
        Executa o algoritmo K-Means até convergência ou atingir max_iter.
        """
        X_arr = np.asarray(X, dtype=float)
        n_samples, n_features = X_arr.shape

        if self.n_clusters > n_samples:
            raise ValueError(f"n_clusters ({self.n_clusters}) não pode ser maior que o número de amostras ({n_samples}).")

        centroids = self._init_centroids(X_arr)
        self.inertia_history_ = []

        for it in range(self.max_iter):
            # 1. Passo de Atribuição (Assignment)
            dist_sq = self._compute_distances(X_arr, centroids)
            labels = np.argmin(dist_sq, axis=1)

            # Cálculo de WCSS / Inércia atual
            min_dist_sq = np.min(dist_sq, axis=1)
            current_inertia = float(np.sum(min_dist_sq))
            self.inertia_history_.append(current_inertia)

            # 2. Passo de Atualização (Update)
            new_centroids = np.empty_like(centroids)
            for k in range(self.n_clusters):
                cluster_mask = (labels == k)
                if np.any(cluster_mask):
                    new_centroids[k] = np.mean(X_arr[cluster_mask], axis=0)
                else:
                    # Trata cluster vazio reatribuindo ao ponto com maior erro
                    furthest_idx = np.argmax(min_dist_sq)
                    new_centroids[k] = X_arr[furthest_idx]

            # 3. Verificação de Convergência
            center_shift = np.sum((new_centroids - centroids) ** 2)
            centroids = new_centroids

            if center_shift <= self.tol:
                self.n_iter_ = it + 1
                break
        else:
            self.n_iter_ = self.max_iter

        # Atualiza atributos finais
        self.cluster_centers_ = centroids
        final_dist_sq = self._compute_distances(X_arr, centroids)
        self.labels_ = np.argmin(final_dist_sq, axis=1)
        self.inertia_ = float(np.sum(np.min(final_dist_sq, axis=1)))

        return self

    def predict(self, X):
        """
        Atribui novos dados ao centroide mais próximo.
        """
        if self.cluster_centers_ is None:
            raise RuntimeError("O modelo K-Means precisa ser ajustado com fit() antes de predict().")
        X_arr = np.asarray(X, dtype=float)
        dist_sq = self._compute_distances(X_arr, self.cluster_centers_)
        return np.argmin(dist_sq, axis=1)

    def fit_predict(self, X):
        """
        Ajusta e retorna os rótulos de cluster.
        """
        return self.fit(X).labels_

    def transform(self, X):
        """
        Transforma X no espaço de distância aos centroides.
        Retorna matriz (n_samples, n_clusters) com distâncias euclidianas.
        """
        if self.cluster_centers_ is None:
            raise RuntimeError("O modelo K-Means precisa ser ajustado com fit() antes de transform().")
        X_arr = np.asarray(X, dtype=float)
        dist_sq = self._compute_distances(X_arr, self.cluster_centers_)
        return np.sqrt(dist_sq)
