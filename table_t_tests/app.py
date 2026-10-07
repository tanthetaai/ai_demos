"""Interactive Streamlit app for two small-sample t-test demonstrations."""

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from scipy import stats

from statistics_demo import (
    NEW_SUPPLIER_MM,
    OLD_SUPPLIER_MM,
    TABLET_WEIGHTS_MG,
    TARGET_WEIGHT_MG,
    supplier_results,
    tablet_batch_results,
)


ALTERNATIVES = {
    "Different from (two-sided)": "two-sided",
    "Less than": "less",
    "Greater than": "greater",
}


def parse_measurements(raw_values: str) -> np.ndarray:
    """Parse comma- or whitespace-separated values and reject invalid entries."""
    tokens = raw_values.replace(",", " ").split()
    if len(tokens) < 2:
        raise ValueError("Enter at least two numeric measurements.")
    try:
        values = np.array([float(token) for token in tokens], dtype=float)
    except ValueError as error:
        raise ValueError("Use only numbers separated by commas or spaces.") from error
    if not np.isfinite(values).all():
        raise ValueError("Measurements must all be finite numbers.")
    return values


def show_t_distribution(
    statistic: float,
    degrees_of_freedom: float,
    alpha: float,
    alternative: str,
) -> None:
    """Draw the null t distribution, rejection tail(s), and observed statistic."""
    distribution = stats.t(df=degrees_of_freedom)
    if alternative == "two-sided":
        critical_value = float(distribution.ppf(1 - alpha / 2))
        x_min = min(-critical_value, statistic) - 1
        x_max = max(critical_value, statistic) + 1
    elif alternative == "less":
        critical_value = float(distribution.ppf(alpha))
        x_min = min(critical_value, statistic) - 1
        x_max = max(2.0, statistic + 1)
    else:
        critical_value = float(distribution.ppf(1 - alpha))
        x_min = min(-2.0, statistic - 1)
        x_max = max(critical_value, statistic) + 1

    x_values = np.linspace(x_min, x_max, 500)
    density = distribution.pdf(x_values)
    figure, axis = plt.subplots(figsize=(8, 3.2))
    axis.plot(x_values, density, color="#176b87", linewidth=2.5)
    if alternative == "two-sided":
        axis.fill_between(
            x_values,
            density,
            where=(x_values <= -critical_value) | (x_values >= critical_value),
            color="#f0b429",
            alpha=0.5,
            label=f"Rejection regions (α={alpha:g})",
        )
        axis.axvline(-critical_value, color="#c05621", linestyle="--")
    elif alternative == "less":
        axis.fill_between(
            x_values,
            density,
            where=x_values <= critical_value,
            color="#f0b429",
            alpha=0.5,
            label=f"Rejection region (α={alpha:g})",
        )
    else:
        axis.fill_between(
            x_values,
            density,
            where=x_values >= critical_value,
            color="#f0b429",
            alpha=0.5,
            label=f"Rejection region (α={alpha:g})",
        )
    axis.axvline(
        statistic,
        color="#b83227",
        linewidth=2,
        label=f"Observed t = {statistic:.2f}",
    )
    axis.axvline(critical_value, color="#c05621", linestyle="--", label="Critical t")
    axis.set(
        title=f"t distribution if the null hypothesis is true (df = {degrees_of_freedom:g})",
        xlabel="t statistic",
        ylabel="Density",
    )
    axis.legend(frameon=False, loc="upper left", fontsize=8)
    axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    st.pyplot(figure, use_container_width=True)
    plt.close(figure)


def code_for_one_sample(
    values: np.ndarray,
    target: float,
    alternative: str,
    confidence_level: float,
) -> str:
    """Build downloadable code matching the active one-sample controls."""
    return f'''import numpy as np
from scipy import stats

weights = np.array({values.tolist()!r})
target_mg = {target!r}
alternative = {alternative!r}
confidence_level = {confidence_level!r}

mean = weights.mean()
sample_sd = weights.std(ddof=1)
test = stats.ttest_1samp(weights, popmean=target_mg, alternative=alternative)
confidence_interval = stats.t.interval(
    confidence_level,
    df=len(weights) - 1,
    loc=mean,
    scale=stats.sem(weights),
)
print(f"mean={{mean:.3f}}, sample SD={{sample_sd:.3f}}")
print(f"t={{test.statistic:.3f}}, df={{test.df:.0f}}, p={{test.pvalue:.5g}}")
print(f"confidence interval={{confidence_interval}}")
'''


def code_for_two_samples(
    old_values: np.ndarray,
    new_values: np.ndarray,
    alternative: str,
) -> str:
    """Build downloadable code matching the active two-sample controls."""
    return f'''import numpy as np
from scipy import stats

old_supplier = np.array({old_values.tolist()!r})
new_supplier = np.array({new_values.tolist()!r})
alternative = {alternative!r}

test = stats.ttest_ind(
    new_supplier,
    old_supplier,
    equal_var=True,
    alternative=alternative,
)
print(f"old mean={{old_supplier.mean():.4f}} mm")
print(f"new mean={{new_supplier.mean():.4f}} mm")
print(f"t={{test.statistic:.3f}}, df={{test.df:.0f}}, p={{test.pvalue:.5g}}")
'''


st.set_page_config(
    page_title="Small Samples on Trial",
    page_icon="⚖️",
    layout="wide",
)

st.markdown(
    """
    <style>
    .stApp { background: #f5f7fa; }
    .hero {
        padding: 1.5rem 1.8rem;
        border-radius: 1rem;
        color: #f8fafc;
        background: linear-gradient(120deg, #102a43, #243b53);
        margin-bottom: 1rem;
    }
    .hero h1 { color: #f8fafc; margin: 0 0 .35rem 0; }
    .hero p { color: #d9e2ec; margin: 0; }
    .lower-third {
        margin-top: 1.2rem;
        padding: .9rem 1.1rem;
        border-left: 5px solid #f0b429;
        border-radius: .35rem;
        background: #102a43;
        color: #fff;
        font-size: 1.15rem;
        font-weight: 700;
    }
    .voiceover {
        color: #334e68;
        border-left: 3px solid #9fb3c8;
        padding: .55rem .8rem;
        margin: .5rem 0 1rem 0;
        font-style: italic;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    """
    <div class="hero">
      <h1>Small Samples on Trial</h1>
      <p>Change the data and hypothesis, then watch the t-test update.</p>
    </div>
    """,
    unsafe_allow_html=True,
)
st.caption("Screen-recording cue · Code in comments")
st.markdown(
    '<div class="voiceover">“Let’s put six tablets on trial.”</div>',
    unsafe_allow_html=True,
)

tablet_tab, supplier_tab = st.tabs(
    [
        "Demo 1 · Is the batch out of spec?",
        "Demo 2 · Old vs. new supplier",
    ]
)

with tablet_tab:
    st.subheader("One-sample t-test")
    data_mode = st.selectbox(
        "Tablet data",
        ["Example measurements", "Enter my own measurements"],
        key="tablet_data_mode",
    )
    if data_mode == "Example measurements":
        tablet_values = TABLET_WEIGHTS_MG.copy()
        st.caption("Example · n = 6 · seed = 8 scenario · measurements are fixed.")
    else:
        raw_weights = st.text_area(
            "Enter tablet weights (mg), separated by commas or spaces",
            value="239.2, 240.9, 240.8, 245.0, 236.8, 245.7",
            key="custom_tablet_weights",
        )
        try:
            tablet_values = parse_measurements(raw_weights)
        except ValueError as error:
            st.error(str(error))
            tablet_values = None

    target = st.number_input(
        "Target weight / null mean μ₀ (mg)",
        value=float(TARGET_WEIGHT_MG),
        step=0.1,
        format="%.2f",
        key="tablet_target",
    )
    alternative_label = st.selectbox(
        "Alternative hypothesis",
        list(ALTERNATIVES),
        key="tablet_alternative",
    )
    alternative = ALTERNATIVES[alternative_label]
    alpha = st.selectbox(
        "Significance level α",
        [0.10, 0.05, 0.01],
        index=1,
        format_func=lambda value: f"{value:g} ({1 - value:.0%} confidence)",
        key="tablet_alpha",
    )

    st.markdown("#### What are we testing?")
    st.latex(r"H_0: \mu = \mu_0")
    alternative_symbols = {
        "two-sided": r"H_1: \mu \ne \mu_0",
        "less": r"H_1: \mu < \mu_0",
        "greater": r"H_1: \mu > \mu_0",
    }
    st.latex(alternative_symbols[alternative])
    st.latex(r"t = \frac{\bar{x} - \mu_0}{s / \sqrt{n}}")

    if tablet_values is not None:
        if len(tablet_values) < 2:
            st.error("A one-sample t-test needs at least two measurements.")
        else:
            result = tablet_batch_results(
                tablet_values,
                target,
                alternative=alternative,
                confidence_level=1 - alpha,
            )
            st.write("Measurements (mg): " + " · ".join(f"{x:g}" for x in tablet_values))
            st.markdown(
                '<div class="voiceover">“Six tablets. Target weight: '
                f'{target:g} milligrams.”</div>',
                unsafe_allow_html=True,
            )
            mean_col, sd_col, t_col, df_col, p_col = st.columns(5)
            mean_col.metric("Sample mean", f"{result['mean']:.2f} mg")
            sd_col.metric("Sample SD", f"{result['sd']:.2f} mg")
            t_col.metric("Observed t", f"{result['t_statistic']:.2f}")
            df_col.metric("Degrees of freedom", f"{result['degrees_of_freedom']:.0f}")
            p_col.metric("p-value", f"{result['p_value']:.4g}")
            st.latex(
                rf"t = \frac{{{result['mean']:.3f} - {target:g}}}"
                rf"{{{result['sd']:.3f}/\sqrt{{{len(tablet_values)}}}}}"
                rf" = {result['t_statistic']:.3f}"
            )
            show_t_distribution(
                result["t_statistic"],
                result["degrees_of_freedom"],
                alpha,
                alternative,
            )
            if result["p_value"] < alpha:
                st.success(
                    f"Reject H₀ at α = {alpha:g}. The data support the "
                    "selected alternative hypothesis."
                )
            else:
                st.info(
                    f"Do not reject H₀ at α = {alpha:g}. The data do not provide "
                    "enough evidence for the selected alternative."
                )

            ci_low, ci_high = result["confidence_interval"]
            st.write(
                f"**{1 - alpha:.0%} confidence interval for μ:** "
                f"[{ci_low:.2f}, {ci_high:.2f}] mg"
            )
            if alternative == "two-sided":
                st.markdown("#### Why not use a z-test?")
                st.write(
                    f"If we incorrectly treat the sample SD as a known population "
                    f"SD, the z approximation gives p = {result['z_p_value']:.2g} "
                    f"instead of {result['p_value']:.4g} from the t-test."
                )
            if data_mode == "Example measurements" and alternative == "two-sided":
                caption = "6 tablets. Still enough to catch a bad batch."
            elif result["p_value"] < alpha:
                caption = "Evidence against the null hypothesis."
            else:
                caption = "Small samples can also tell you: not yet."
            st.markdown(
                f'<div class="lower-third">{caption}</div>',
                unsafe_allow_html=True,
            )

            tablet_code = code_for_one_sample(
                tablet_values, target, alternative, 1 - alpha
            )
            st.download_button(
                "Download code for these inputs",
                data=tablet_code,
                file_name="one_sample_t_test.py",
                mime="text/x-python",
                key="download_tablet_code",
            )
            with st.expander("Code in comments · Demo 1"):
                st.code(tablet_code, language="python")

with supplier_tab:
    st.subheader("Independent two-sample t-test")
    data_mode = st.selectbox(
        "Supplier data",
        ["Example measurements", "Enter my own measurements"],
        key="supplier_data_mode",
    )
    if data_mode == "Example measurements":
        old_values = OLD_SUPPLIER_MM.copy()
        new_values = NEW_SUPPLIER_MM.copy()
        st.caption("Example · five bolts per supplier · diameters in millimetres.")
    else:
        old_col, new_col = st.columns(2)
        with old_col:
            raw_old = st.text_area(
                "Current supplier values (mm)",
                value=", ".join(f"{value:g}" for value in OLD_SUPPLIER_MM),
                key="custom_old_supplier",
            )
        with new_col:
            raw_new = st.text_area(
                "Candidate supplier values (mm)",
                value=", ".join(f"{value:g}" for value in NEW_SUPPLIER_MM),
                key="custom_new_supplier",
            )
        try:
            old_values = parse_measurements(raw_old)
            new_values = parse_measurements(raw_new)
        except ValueError as error:
            st.error(str(error))
            old_values = None
            new_values = None

    alternative_label = st.selectbox(
        "Alternative hypothesis (new minus current)",
        list(ALTERNATIVES),
        key="supplier_alternative",
    )
    alternative = ALTERNATIVES[alternative_label]
    alpha = st.selectbox(
        "Significance level α",
        [0.10, 0.05, 0.01],
        index=1,
        format_func=lambda value: f"{value:g} ({1 - value:.0%} confidence)",
        key="supplier_alpha",
    )
    st.markdown("#### What are we testing?")
    st.latex(r"H_0: \mu_{\mathrm{new}} = \mu_{\mathrm{current}}")
    supplier_alternatives = {
        "two-sided": r"H_1: \mu_{\mathrm{new}} \ne \mu_{\mathrm{current}}",
        "less": r"H_1: \mu_{\mathrm{new}} < \mu_{\mathrm{current}}",
        "greater": r"H_1: \mu_{\mathrm{new}} > \mu_{\mathrm{current}}",
    }
    st.latex(supplier_alternatives[alternative])
    st.latex(
        r"t = \frac{\bar{x}_{\mathrm{new}} - \bar{x}_{\mathrm{current}}}"
        r"{s_p \sqrt{1/n_{\mathrm{new}} + 1/n_{\mathrm{current}}}}"
    )

    if old_values is not None and new_values is not None:
        result = supplier_results(old_values, new_values, alternative=alternative)
        old_col, new_col = st.columns(2)
        old_col.metric("Current supplier mean", f"{result['old_mean']:.4f} mm")
        new_col.metric("Candidate supplier mean", f"{result['new_mean']:.4f} mm")
        st.write(
            "Current: "
            + " · ".join(f"{x:g}" for x in old_values)
            + " mm"
        )
        st.write(
            "Candidate: "
            + " · ".join(f"{x:g}" for x in new_values)
            + " mm"
        )
        t_col, df_col, p_col = st.columns(3)
        t_col.metric("Observed t", f"{result['t_statistic']:.2f}")
        df_col.metric("Degrees of freedom", f"{result['degrees_of_freedom']:.0f}")
        p_col.metric("p-value", f"{result['p_value']:.4g}")
        st.latex(
            rf"t = \frac{{{result['new_mean']:.4f} - {result['old_mean']:.4f}}}"
            rf"{{{result['standard_error']:.5f}}}"
            rf" = {result['t_statistic']:.3f}"
        )
        show_t_distribution(
            result["t_statistic"],
            result["degrees_of_freedom"],
            alpha,
            alternative,
        )
        if result["p_value"] < alpha:
            st.success(
                f"Reject H₀ at α = {alpha:g}. The data support the selected "
                "supplier difference."
            )
        else:
            st.info(
                f"Do not reject H₀ at α = {alpha:g}. The data do not provide "
                "enough evidence for the selected supplier difference."
            )
        if data_mode == "Example measurements" and alternative == "two-sided":
            caption = "Small samples can also tell you: not yet."
        elif result["p_value"] < alpha:
            caption = "Evidence against equal supplier means."
        else:
            caption = "Small samples can also tell you: not yet."
        st.markdown(
            f'<div class="lower-third">{caption}</div>',
            unsafe_allow_html=True,
        )

        supplier_code = code_for_two_samples(old_values, new_values, alternative)
        st.download_button(
            "Download code for these inputs",
            data=supplier_code,
            file_name="two_sample_t_test.py",
            mime="text/x-python",
            key="download_supplier_code",
        )
        with st.expander("Code in comments · Demo 2"):
            st.code(supplier_code, language="python")

st.caption(
    "Choose example data or enter your own measurements. The hypothesis, "
    "t-statistic, p-value, and null-distribution plot update with your selections."
)
