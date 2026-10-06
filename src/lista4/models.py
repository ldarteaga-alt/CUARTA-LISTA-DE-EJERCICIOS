import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression

def fit_linear(df, predictors, target="y"):
    """Ajusta una regresión lineal con intercepto."""
    model = LinearRegression(fit_intercept=True)
    model.fit(df[predictors], df[target])
    return model

def linear_predictions(model, df, predictors):
    return model.predict(df[predictors])

def fit_logistic(df, predictors, target="abandona_30d"):
    """Ajusta regresión logística según las especificaciones de la lista."""
    model = LogisticRegression(
        solver="lbfgs",
        C=1e6,
        max_iter=5000
    )
    model.fit(df[predictors], df[target])
    return model

def logistic_probabilities(model, df, predictors):
    return model.predict_proba(df[predictors])[:, 1]

def logistic_classes(model, df, predictors, threshold=0.5):
    probabilities = logistic_probabilities(model, df, predictors)
    return (probabilities >= threshold).astype(int)
