from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from src.napsu_mq.logistic_regression import logistic_regression


def create_model():
    return LogisticRegression(random_state=0)


def run_logistic_regression(df_np, X_train, y_train, X_test, y_test):
    model = create_model()
    model.fit(X_train, y_train)

    accuracy_score = cross_val_score(model, X_test, y_test, scoring='accuracy', n_jobs=-1, error_score='raise')
    balanced_accuracy_score = cross_val_score(model, X_test, y_test, scoring='balanced_accuracy', n_jobs=-1,
                                              error_score='raise')
    f1_score = cross_val_score(model, X_test, y_test, scoring='f1', n_jobs=-1, error_score='raise')

    point_estimates, variance_estimates, confidence_intervals = logistic_regression(
        df_np, add_constant=False, return_intervals=True, conf_levels=[0.95])

    coefficients = model.coef_

    return accuracy_score, balanced_accuracy_score, f1_score, coefficients, point_estimates, variance_estimates, confidence_intervals
