"""
Pacote core do Projeto 02 - IA 2026.
Implementações 100% from scratch em NumPy e Pandas.
"""

from .preprocessing import (
    train_test_split,
    StandardScaler,
    MinMaxScaler,
    LabelEncoder,
    impute_missing,
    encode_categorical_features
)

from .metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    wcss,
    silhouette_score
)

from .decision_tree import DecisionTreeClassifier, Node
from .kmeans import KMeans
from .apriori import apriori, association_rules
