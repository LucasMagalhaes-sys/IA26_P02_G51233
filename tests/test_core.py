"""
Testes unitários automatizados para os módulos core (FASE 1).
Valida ausência de bibliotecas proibidas, convergência e hiperparâmetros.
"""

import sys
import os
import unittest
import numpy as np
import pandas as pd

# Adicionar raiz ao path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.preprocessing import (
    train_test_split, StandardScaler, MinMaxScaler, LabelEncoder, impute_missing
)
from src.metrics import (
    confusion_matrix, accuracy_score, precision_score, recall_score, f1_score,
    wcss, silhouette_score
)
from src.decision_tree import DecisionTreeClassifier
from src.kmeans import KMeans
from src.apriori import apriori, association_rules


class TestForbiddenImports(unittest.TestCase):
    """Garante que nenhum módulo em src/ importa sklearn ou libs proibidas."""
    def test_no_prohibited_imports_in_src(self):
        src_dir = os.path.join(BASE_DIR, "src")
        prohibited = ["sklearn", "scipy.cluster", "torch", "tensorflow", "statsmodels", "xgboost", "lightgbm"]
        for fname in os.listdir(src_dir):
            if fname.endswith(".py"):
                fpath = os.path.join(src_dir, fname)
                with open(fpath, "r", encoding="utf-8") as f:
                    content = f.read()
                    for p in prohibited:
                        self.assertNotIn(f"import {p}", content, f"Import proibido '{p}' encontrado em {fname}!")
                        self.assertNotIn(f"from {p}", content, f"Import proibido '{p}' encontrado em {fname}!")


class TestPreprocessingAndMetrics(unittest.TestCase):
    def test_train_test_split_stratified(self):
        X = np.arange(100).reshape(100, 1)
        y = np.array([0] * 80 + [1] * 20)
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        self.assertEqual(len(X_tr), 80)
        self.assertEqual(len(X_te), 20)
        # Proporção mantida: 80% de 0 e 20% de 1
        self.assertAlmostEqual(np.mean(y_te == 1), 0.20, places=2)

    def test_standard_scaler(self):
        data = np.array([[10.0, 2.0], [20.0, 4.0], [30.0, 6.0]])
        scaler = StandardScaler()
        scaled = scaler.fit_transform(data)
        np.testing.assert_almost_equal(np.mean(scaled, axis=0), [0.0, 0.0], decimal=5)
        np.testing.assert_almost_equal(np.std(scaled, axis=0), [1.0, 1.0], decimal=5)
        inv = scaler.inverse_transform(scaled)
        np.testing.assert_almost_equal(inv, data, decimal=5)

    def test_classification_metrics(self):
        y_true = np.array([0, 1, 0, 1, 0, 0, 1, 1])
        y_pred = np.array([0, 1, 0, 0, 0, 1, 1, 1])
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        # TP=3, FP=1, TN=3, FN=1
        self.assertEqual(cm[1, 1], 3)
        self.assertEqual(cm[0, 1], 1)
        self.assertEqual(accuracy_score(y_true, y_pred), 6 / 8)
        self.assertAlmostEqual(precision_score(y_true, y_pred, pos_label=1), 3 / 4)
        self.assertAlmostEqual(recall_score(y_true, y_pred, pos_label=1), 3 / 4)
        self.assertAlmostEqual(f1_score(y_true, y_pred, pos_label=1), 3 / 4)


class TestDecisionTreeClassifier(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        # Dataset 2D linearmente separável
        c0 = np.random.randn(60, 2) + np.array([-3.0, -3.0])
        c1 = np.random.randn(60, 2) + np.array([3.0, 3.0])
        self.X = np.vstack([c0, c1])
        self.y = np.array([0] * 60 + [1] * 60)

    def test_convergence_and_accuracy(self):
        dt = DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=42)
        dt.fit(self.X, self.y)
        acc = dt.score(self.X, self.y)
        self.assertGreaterEqual(acc, 0.95, f"Árvore deveria ter alta acurácia em dados separáveis, obteve {acc}")
        self.assertIsNotNone(dt.feature_importances_)
        self.assertAlmostEqual(np.sum(dt.feature_importances_), 1.0, places=4)

    def test_criterion_entropy(self):
        dt = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=42)
        dt.fit(self.X, self.y)
        acc = dt.score(self.X, self.y)
        self.assertGreaterEqual(acc, 0.95)

    def test_max_depth_stump(self):
        dt = DecisionTreeClassifier(max_depth=1, random_state=42)
        dt.fit(self.X, self.y)
        # Root node tem depth 0, com max_depth 1 seus filhos são folhas
        self.assertFalse(dt.root.is_leaf)
        self.assertTrue(dt.root.left.is_leaf)
        self.assertTrue(dt.root.right.is_leaf)

    def test_min_samples_leaf_constraint(self):
        dt = DecisionTreeClassifier(max_depth=5, min_samples_leaf=15, random_state=42)
        dt.fit(self.X, self.y)
        # Verificar recursivamente que todas as folhas têm >= 15 amostras
        def check_leaves(node):
            if node.is_leaf:
                self.assertGreaterEqual(node.n_samples, 15)
            else:
                check_leaves(node.left)
                check_leaves(node.right)
        check_leaves(dt.root)


class TestKMeans(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        # 3 clusters bem definidos
        c0 = np.random.randn(50, 2) + np.array([-10.0, 0.0])
        c1 = np.random.randn(50, 2) + np.array([0.0, 10.0])
        c2 = np.random.randn(50, 2) + np.array([10.0, 0.0])
        self.X = np.vstack([c0, c1, c2])

    def test_kmeans_convergence_and_inertia(self):
        km = KMeans(n_clusters=3, init='k-means++', random_state=42)
        km.fit(self.X)
        self.assertEqual(len(np.unique(km.labels_)), 3)
        self.assertGreater(len(km.inertia_history_), 1)
        # Inércia final deve ser menor ou igual à inicial
        self.assertLessEqual(km.inertia_history_[-1], km.inertia_history_[0])
        # WCSS calculado confere com km.inertia_
        calc_wcss = wcss(self.X, km.labels_, km.cluster_centers_)
        self.assertAlmostEqual(calc_wcss, km.inertia_, places=3)

    def test_silhouette_score_high_for_separated(self):
        km = KMeans(n_clusters=3, random_state=42)
        km.fit(self.X)
        sil = silhouette_score(self.X, km.labels_)
        self.assertGreater(sil, 0.70, f"Clusters bem separados devem ter silhouette > 0.70, obtido: {sil}")

    def test_reproducibility(self):
        km1 = KMeans(n_clusters=3, random_state=42).fit(self.X)
        km2 = KMeans(n_clusters=3, random_state=42).fit(self.X)
        np.testing.assert_array_equal(km1.labels_, km2.labels_)
        self.assertAlmostEqual(km1.inertia_, km2.inertia_, places=5)


class TestApriori(unittest.TestCase):
    def test_apriori_and_rules(self):
        data = pd.DataFrame([
            {'leite': 1, 'pao': 1, 'manteiga': 1, 'cafe': 0},
            {'leite': 1, 'pao': 0, 'manteiga': 1, 'cafe': 1},
            {'leite': 0, 'pao': 1, 'manteiga': 1, 'cafe': 1},
            {'leite': 1, 'pao': 1, 'manteiga': 1, 'cafe': 1},
            {'leite': 0, 'pao': 1, 'manteiga': 0, 'cafe': 0},
        ])
        itemsets = apriori(data, min_support=0.4, use_colnames=True)
        self.assertFalse(itemsets.empty)
        rules = association_rules(itemsets, metric='confidence', min_threshold=0.6)
        self.assertFalse(rules.empty)
        self.assertIn('confidence', rules.columns)
        self.assertIn('lift', rules.columns)


if __name__ == '__main__':
    unittest.main()
