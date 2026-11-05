"""LLMetrics Classification Package."""

from core.classification.logistic import fit_binary_logistic_regression
from core.classification.multinomial import fit_multinomial_logistic_regression

__all__ = ["fit_binary_logistic_regression", "fit_multinomial_logistic_regression"]
