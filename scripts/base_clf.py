from typing import List, Protocol
import numpy as np
from sklearn.model_selection import cross_val_score
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
        MLPClassifier(hidden_layer_sizes=(100), solver='sgd', alpha=0.5)
    ]

    return models


def run_classification_on_adult(train_df, test_df, target_column):
    models = create_models()

    X_train, y_train = train_df.drop(columns=[target_column], axis=1), train_df[target_column]
    X_test, y_test = test_df.drop(columns=[target_column], axis=1), test_df[target_column]

    scores = []

    for model in models:
        model.fit(X_train, y_train)

        accuracy_score = cross_val_score(model, X_test, y_test, scoring='accuracy', n_jobs=-1, error_score='raise')
        balanced_accuracy_score = cross_val_score(model, X_test, y_test, scoring='balanced_accuracy', n_jobs=-1,
                                                  error_score='raise')

        f1_score = cross_val_score(model, X_test, y_test, scoring='f1', n_jobs=-1, error_score='raise')

        scores.append((type(model).__name__, accuracy_score, balanced_accuracy_score, f1_score))

        print(
            f'Model: {type(model).__name__} \t Accuracy: {np.mean(accuracy_score):.3f} ({np.std(accuracy_score):.3f}), Balanced accuracy: {np.mean(balanced_accuracy_score):.3f} ({np.std(balanced_accuracy_score):.3f}), F1: {np.mean(f1_score):.3f} ({np.std(f1_score):.3f})')

    return scores
