import warnings

from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score
from src.napsu_mq.logistic_regression import logistic_regression, logistic_regression_on_2d
from functools import partial
import statsmodels.api as sm
from sklearn.base import BaseEstimator, RegressorMixin

class SMWrapper(BaseEstimator, RegressorMixin):
    """
        From https://stackoverflow.com/questions/41045752/using-statsmodel-estimations-with-scikit-learn-cross-validation-is-it-possible
        A universal sklearn-style wrapper for statsmodels regressors
    """

    def __init__(self, model_class, fit_intercept=True):
        self.model_class = model_class
        self.fit_intercept = fit_intercept
        self.results = None
        self.model = None
        self.fitted = False

    def fit(self, X, y):
        if self.fit_intercept is True:
            X = sm.add_constant(X)
        self.model = self.model_class(y, X)
        self.results = self.model.fit()
        self.fitted = True
        return self

    def predict(self, X):
        if self.fit_intercept is True:
            X = sm.add_constant(X)
        return self.results.predict(X)

    def get_clf(self):
        if self.fitted is False:
            warnings.warn("Classifier is not fitted yet!")
        return self.results


def create_model():
    binomial_GLM = partial(sm.GLM, family=sm.families.Binomial())
    return SMWrapper(binomial_GLM)


def run_logistic_regression_on_3d(df_np, X_train, y_train, X_test, y_test, col_to_predict: int):
    model = create_model()
    model.fit(X_train, y_train)

    probabilities = model.predict(X_test)
    predictions = probabilities >= 0.5

    accuracy = accuracy_score(y_test, predictions)
    balanced_accuracy = balanced_accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)

    point_estimates, variance_estimates = logistic_regression(
        df_np, col_to_predict=col_to_predict, add_constant=False, return_intervals=False)

    coefficients = model.get_clf().params.to_numpy()

    return accuracy, balanced_accuracy, f1, coefficients, point_estimates, variance_estimates


def run_logistic_regression_on_2d(df_np, X_train, y_train, X_test, y_test, col_to_predict: int, return_confidence_intervals=False):
    model = create_model()
    model.fit(X_train, y_train)

    probabilities = model.predict(X_test)
    predictions = probabilities >= 0.5

    accuracy = accuracy_score(y_test, predictions)
    balanced_accuracy = balanced_accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)

    coefficients = model.get_clf().params.to_numpy()


    if return_confidence_intervals is True:
        point_estimates, variance_estimates, confidence_interval = logistic_regression(
            df_np, col_to_predict=col_to_predict, add_constant=False, return_intervals=True, conf_levels=[0.95])
        return accuracy, balanced_accuracy, f1, coefficients, point_estimates, variance_estimates, confidence_interval

    else:
        point_estimates, variance_estimates = logistic_regression_on_2d(
            df_np, col_to_predict=col_to_predict, add_constant=False, return_intervals=False)
        return accuracy, balanced_accuracy, f1, coefficients, point_estimates, variance_estimates
