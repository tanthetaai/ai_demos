"""Statistical calculations used by the Streamlit t-test demonstrations."""

import numpy as np
from scipy import stats


TABLET_WEIGHTS_MG = np.array([239.2, 240.9, 240.8, 245.0, 236.8, 245.7])
TARGET_WEIGHT_MG = 250.0

OLD_SUPPLIER_MM = np.array([11.9896, 12.0192, 12.0686, 12.1180, 12.1476])
NEW_SUPPLIER_MM = np.array([12.0426, 12.0722, 12.1216, 12.1710, 12.2006])


def tablet_batch_results(
    values: np.ndarray,
    target: float,
    alternative: str = "two-sided",
    confidence_level: float = 0.95,
) -> dict[str, float | tuple[float, float]]:
    """Compute a one-sample t-test, CI, and intentionally wrong z comparison."""
    mean = float(values.mean())
    sample_sd = float(values.std(ddof=1))
    test = stats.ttest_1samp(values, popmean=target, alternative=alternative)
    interval = stats.t.interval(
        confidence_level,
        df=len(values) - 1,
        loc=mean,
        scale=stats.sem(values),
    )
    z_statistic = (mean - target) / (sample_sd / np.sqrt(len(values)))

    return {
        "mean": mean,
        "sd": sample_sd,
        "t_statistic": float(test.statistic),
        "p_value": float(test.pvalue),
        "degrees_of_freedom": float(test.df),
        "confidence_interval": (float(interval[0]), float(interval[1])),
        "z_p_value": float(2 * stats.norm.sf(abs(z_statistic))),
        "standard_error": float(stats.sem(values)),
    }


def supplier_results(
    old_values: np.ndarray,
    new_values: np.ndarray,
    alternative: str = "two-sided",
) -> dict[str, float]:
    """Compute the equal-variance, independent two-sample t-test."""
    test = stats.ttest_ind(
        new_values,
        old_values,
        equal_var=True,
        alternative=alternative,
    )
    return {
        "old_mean": float(old_values.mean()),
        "new_mean": float(new_values.mean()),
        "t_statistic": float(test.statistic),
        "degrees_of_freedom": float(test.df),
        "p_value": float(test.pvalue),
        "standard_error": float(
            np.sqrt(
                (
                    (len(old_values) - 1) * old_values.var(ddof=1)
                    + (len(new_values) - 1) * new_values.var(ddof=1)
                )
                / (len(old_values) + len(new_values) - 2)
                * (1 / len(old_values) + 1 / len(new_values))
            )
        ),
    }
