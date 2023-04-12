from typing import Protocol
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.dummy import DummyClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


class ScikitModel(Protocol):
    def fit(self, X, y, sample_weight=None) -> 'ScikitModel': ...

    def predict(self, X) -> np.ndarray: ...

    def score(self, X, y, sample_weight=None) -> float: ...

    def set_params(self, **params) -> 'ScikitModel': ...


def create_models():
    models = [
        DummyClassifier(strategy="most_frequent"),
        GradientBoostingClassifier(),
        LGBMClassifier(),
        XGBClassifier(),
        RandomForestClassifier(),
        LinearSVC(),
        MLPClassifier(hidden_layer_sizes=100, solver='sgd', alpha=0.5)
    ]

    return models


def run_classification(train_df, test_df, target_column):
    models = create_models()

    X_train, y_train = train_df.drop(columns=[target_column]), train_df[target_column]
    X_test, y_test = test_df.drop(columns=[target_column]), test_df[target_column]

    if len(y_train.unique()) == 1:
        # If there is only one class, the dataset is not suitable for classification
        return None

    scores = []

    for model in models:
        model = model.fit(X_train, y_train)

        probabilities = model.predict(X_test)
        predictions = probabilities >= 0.5

        accuracy = accuracy_score(y_test, predictions)
        balanced_accuracy = balanced_accuracy_score(y_test, predictions)
        f1 = f1_score(y_test, predictions)

        scores.append((type(model).__name__, accuracy, balanced_accuracy, f1))

        # print(
        #    f'Model: {type(model).__name__} \t Accuracy: {np.mean(accuracy_score):.3f} ({np.std(accuracy_score):.3f}), Balanced accuracy: {np.mean(balanced_accuracy_score):.3f} ({np.std(balanced_accuracy_score):.3f}), F1: {np.mean(f1_score):.3f} ({np.std(f1_score):.3f})')

    return scores
