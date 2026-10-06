import numpy as np

def empirical_risk(y_true, y_pred):
    """Riesgo empírico con pérdida cuadrática."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return np.mean((y_true - y_pred) ** 2)

def logistic_risk(y_true, probabilities):
    """Riesgo empírico logístico (log-loss)."""
    y_true = np.asarray(y_true)
    probabilities = np.asarray(probabilities)
    probabilities = np.clip(probabilities, 1e-15, 1 - 1e-15)
    return -np.mean(
        y_true * np.log(probabilities)
        + (1 - y_true) * np.log(1 - probabilities)
    )

def classification_errors(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return int(np.sum(y_true != y_pred))

def classification_error_rate(y_true, y_pred):
    return classification_errors(y_true, y_pred) / len(y_true)

def absolute_prediction_difference(pred1, pred2):
    return np.abs(np.asarray(pred1) - np.asarray(pred2))
