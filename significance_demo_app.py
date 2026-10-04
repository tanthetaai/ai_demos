"""Portrait Streamlit app for interactive demonstrations of statistical significance."""

from math import comb, erfc, sqrt

import streamlit as st


st.set_page_config(
    page_title="Is that result real?",
    page_icon="🔎",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    .stApp {
        background: #e9eff0;
        color: #203746;
    }
    [data-testid="stHeader"] {
        background: transparent;
    }
    [data-testid="stMainBlockContainer"] {
        max-width: 460px;
        min-height: 100vh;
        padding: 1.1rem 1.1rem 2rem;
        background: #fbfcfa;
        box-shadow: 0 0 35px #18344118;
    }
    h1, h2, h3, p, label {
        font-family: Inter, ui-sans-serif, system-ui, sans-serif;
    }
    .reel-header {
        padding: 22px 21px;
        border-radius: 21px;
        color: white;
        background: linear-gradient(135deg, #102a43, #174c5e 58%, #287c78);
        margin: 4px 0 18px;
    }
    .reel-kicker {
        font-size: 10px;
        letter-spacing: 1.8px;
        color: #a8e3d6;
        font-weight: 750;
        text-transform: uppercase;
    }
    .reel-header h1 {
        color: white;
        font-size: 28px;
        line-height: 1.12;
        margin: 9px 0 8px;
    }
    .reel-header p {
        color: #e4f2f0;
        font-size: 14px;
        line-height: 1.5;
        margin: 0;
    }
    .step {
        color: #173d4b;
        font-size: 16px;
        font-weight: 750;
        margin: 18px 0 4px;
    }
    .hint {
        color: #647884;
        font-size: 12px;
        line-height: 1.5;
        margin: 0 0 10px;
    }
    .result-card {
        background: white;
        border: 1px solid #e2eaed;
        border-radius: 15px;
        padding: 13px 15px;
        margin: 9px 0;
        box-shadow: 0 4px 12px #14384a0a;
    }
    .result-label {
        color: #627784;
        font-size: 10px;
        letter-spacing: .8px;
        text-transform: uppercase;
    }
    .result-value {
        color: #173d4b;
        font-size: 24px;
        font-weight: 780;
        margin-top: 5px;
    }
    .result-detail {
        color: #637681;
        font-size: 12px;
        line-height: 1.45;
        margin-top: 3px;
    }
    .callout {
        background: #eff8f5;
        border-left: 4px solid #287c78;
        border-radius: 0 11px 11px 0;
        line-height: 1.5;
        margin: 12px 0;
        padding: 12px 14px;
        font-size: 13px;
    }
    .callout.warn {
        background: #fff8e9;
        border-color: #dc9941;
    }
    .callout.bad {
        background: #fff2ef;
        border-color: #d26a60;
    }
    .cup-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 7px;
        margin: 12px 0;
    }
    .cup {
        background: white;
        border: 1px solid #dce7e9;
        border-radius: 12px;
        padding: 9px 2px;
        text-align: center;
    }
    .cup span {
        display: block;
        font-size: 23px;
    }
    .cup small {
        color: #60747d;
        font-size: 9px;
    }
    .p-value {
        display: inline-block;
        background: #eef3f5;
        border-radius: 8px;
        color: #3d5662;
        font-size: 11px;
        margin: 3px 2px 3px 0;
        padding: 6px 8px;
    }
    .p-value.win {
        background: #ffede8;
        color: #aa4237;
        font-weight: 750;
    }
    .fine-print {
        color: #74858c;
        font-size: 10px;
        line-height: 1.5;
        margin-top: 12px;
    }
    div[data-testid="stSlider"] {
        padding: 0 .1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="reel-header">
      <div class="reel-kicker">A 9:16 interactive stats demo</div>
      <h1>Is that result real?</h1>
      <p>Change a setting. See what the evidence says. No code or statistics background needed.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


def show_card(label: str, value: str, detail: str) -> None:
    st.markdown(
        f'<div class="result-card"><div class="result-label">{label}</div>'
        f'<div class="result-value">{value}</div>'
        f'<div class="result-detail">{detail}</div></div>',
        unsafe_allow_html=True,
    )


def show_callout(text: str, tone: str = "") -> None:
    st.markdown(
        f'<div class="callout {tone}">{text}</div>',
        unsafe_allow_html=True,
    )


def two_sided_p(z_score: float) -> float:
    return erfc(abs(z_score) / sqrt(2))


def compare_rates(rate_a: float, rate_b: float, sample_size: int) -> dict[str, float | int]:
    """Calculate a pooled two-proportion z-test and an unpooled Wald interval."""
    conversions_a = round(rate_a * sample_size / 100)
    conversions_b = round(rate_b * sample_size / 100)
    observed_a = conversions_a / sample_size
    observed_b = conversions_b / sample_size
    difference = observed_b - observed_a
    pooled_rate = (conversions_a + conversions_b) / (2 * sample_size)
    pooled_se = sqrt(pooled_rate * (1 - pooled_rate) * 2 / sample_size)
    z_score = difference / pooled_se if pooled_se else 0.0
    p_value = two_sided_p(z_score)
    interval_se = sqrt(
        observed_a * (1 - observed_a) / sample_size
        + observed_b * (1 - observed_b) / sample_size
    )
    return {
        "conversions_a": conversions_a,
        "conversions_b": conversions_b,
        "rate_a": observed_a,
        "rate_b": observed_b,
        "difference": difference,
        "pooled_rate": pooled_rate,
        "z_score": z_score,
        "p_value": p_value,
        "low": difference - 1.96 * interval_se,
        "high": difference + 1.96 * interval_se,
        "interval_se": interval_se,
    }


def show_ab_result(rate_a: float, rate_b: float, sample_size: int, title: str) -> None:
    result = compare_rates(rate_a, rate_b, sample_size)
    low = float(result["low"])
    high = float(result["high"])
    p_value = float(result["p_value"])
    difference = float(result["difference"])
    interval_digits = 2 if sample_size >= 100_000 else 1

    st.markdown(f"**{title}**")
    show_card(
        "Version A",
        f'{float(result["rate_a"]) * 100:.2f}%',
        f'{int(result["conversions_a"]):,} of {sample_size:,} visitors',
    )
    show_card(
        "Version B",
        f'{float(result["rate_b"]) * 100:.2f}%',
        f'{int(result["conversions_b"]):,} of {sample_size:,} visitors',
    )
    show_card(
        "Observed change · B minus A",
        f"{difference * 100:+.2f} pp",
        "pp means percentage points, not percent change.",
    )
    show_card(
        "Chance result this extreme if rates are equal",
        f"{p_value:.6g}",
        "Two-sided p-value. Smaller means harder to explain by chance alone.",
    )
    show_card(
        "95% confidence interval for the change",
        f'{low * 100:+.{interval_digits}f} to {high * 100:+.{interval_digits}f} pp',
        "If zero is inside this range, no difference is still plausible.",
    )

    if p_value < 0.05 and low > 0:
        show_callout(
            "<b>What this means:</b> Evidence supports a positive lift in B. "
            "This does not automatically mean the lift is valuable enough to ship."
        )
    elif p_value < 0.05 and high < 0:
        show_callout(
            "<b>What this means:</b> Evidence suggests B performs worse than A.",
            "warn",
        )
    else:
        show_callout(
            "<b>What this means:</b> The interval still includes zero. "
            "We do not have clear evidence of a real difference yet.",
            "warn",
        )
    st.caption(
        f"Calculation check: pooled conversion rate "
        f'{float(result["pooled_rate"]) * 100:.3f}%; '
        f'z-score {float(result["z_score"]):.3f}.'
    )


demo = st.selectbox(
    "Choose a demo",
    (
        "1 · The tea tasting",
        "2 · Same lift, more data",
        "3 · Does the lift matter?",
        "4 · The multiple-testing trap",
    ),
    label_visibility="collapsed",
)
st.progress(
    (1 + ("2 ·" in demo) + ("3 ·" in demo) + ("4 ·" in demo)) / 4,
    text="Pick a demo · results update as you change controls",
)

if demo.startswith("1"):
    st.subheader("Can she really tell?")
    st.caption(
        "Four cups had milk poured first and four had tea poured first. "
        "The order was hidden."
    )
    cups = "".join(
        f'<div class="cup"><span>☕</span><small>Blind cup {index}</small></div>'
        for index in (4, 8, 2, 7, 1, 6, 3, 5)
    )
    st.markdown(f'<div class="cup-grid">{cups}</div>', unsafe_allow_html=True)
    score = st.slider(
        "How many cups did she identify correctly?",
        min_value=0,
        max_value=4,
        value=4,
        help="Choose the score to see how often a guesser would do this well or better.",
    )
    favorable = sum(comb(4, correct) * comb(4, 4 - correct) for correct in range(score, 5))
    chance = favorable / comb(8, 4)
    show_card("Possible ways to arrange the four milk-first cups", "70", "This stays fixed: C(8, 4).")
    show_card(
        "Ways that score this well or better",
        str(favorable),
        f"{score} or more cups correct.",
    )
    show_card(
        "Chance if she is only guessing",
        f"{favorable}/70 = {chance:.2%}",
        "This is the chance of this score or a better one.",
    )
    if score == 4:
        show_callout(
            "<b>Verdict:</b> A perfect score happens in only 1 of 70 arrangements. "
            "That is unusual enough to question pure guessing."
        )
    elif score == 3:
        show_callout(
            "<b>Verdict:</b> One wrong cup changes the story: 3 correct or better "
            "happens about 24% of the time by guessing. Not convincing by itself.",
            "warn",
        )
    else:
        show_callout(
            "<b>Verdict:</b> A guesser would often do at least this well. "
            "This score is weak evidence that she can tell.",
            "warn",
        )
    st.caption(
        "Why count “this well or better”? A 3/4 is more surprising than 2/4, "
        "so include outcomes at least as strong as the observed score. Exact count; no simulation."
    )

elif demo.startswith("2"):
    st.subheader("Same effect. More data.")
    st.caption(
        "Adjust conversion rates. We compare the same rates with 1,000 and 10,000 visitors per version."
    )
    rate_a = st.slider("A conversion rate (%)", 0.0, 30.0, 10.0, 0.1)
    rate_b = st.slider("B conversion rate (%)", 0.0, 30.0, 12.0, 0.1)
    show_card("Difference you set · B minus A", f"{rate_b - rate_a:+.1f} pp", "Same effect in both tests.")
    st.divider()
    show_ab_result(rate_a, rate_b, 1_000, "SMALL TEST · 1,000 people per version")
    st.divider()
    show_ab_result(rate_a, rate_b, 10_000, "LARGE TEST · 10,000 people per version")
    show_callout(
        "<b>Takeaway:</b> More data usually narrows the confidence interval. "
        "The lift did not change—only how clearly we can distinguish it from zero."
    )

elif demo.startswith("3"):
    st.subheader("Significant — but does it matter?")
    st.caption("Change the rates and sample size. Then judge the effect in business terms.")
    rate_a = st.slider("A conversion rate (%)", 0.0, 30.0, 10.0, 0.01)
    rate_b = st.slider("B conversion rate (%)", 0.0, 30.0, 10.1, 0.01)
    sample_size = st.select_slider(
        "People in each version",
        options=(20_000, 200_000, 2_000_000),
        value=2_000_000,
        format_func=lambda value: f"{value:,}",
    )
    show_ab_result(rate_a, rate_b, sample_size, "YOUR RESULT")
    extra_per_thousand = (rate_b - rate_a) / 100 * 1_000
    show_callout(
        f"<b>Business translation:</b> At these rates, the change is about "
        f"{extra_per_thousand:g} additional conversion per 1,000 visitors. "
        "Statistical significance does not tell us whether those conversions "
        "are worth engineering, maintenance, and opportunity costs.",
        "warn",
    )
    st.caption(
        "Ask: What is a conversion worth? What will the change cost? "
        "Would it still be worthwhile near the low end of the interval?"
    )

else:
    st.subheader("If you test enough, something may win")
    st.caption(
        "A/A means both versions are identical. The p-values below are fixed teaching examples."
    )
    p_values = [
        0.380, 0.710, 0.022, 0.640, 0.190, 0.830, 0.110, 0.570, 0.920, 0.340,
        0.270, 0.760, 0.480, 0.680, 0.130, 0.041, 0.590, 0.870, 0.210, 0.004,
    ]
    number_of_tests = st.slider("Number of tests to look at", 1, 20, 20)
    displayed = p_values[:number_of_tests]
    false_wins = sum(value < 0.05 for value in displayed)
    familywise = 1 - (1 - 0.05) ** number_of_tests
    show_card("Identical versions", "10%", "True conversion rate in both versions.")
    show_card("Apparently significant tests", f"{false_wins} of {number_of_tests}", "p-value below 0.05.")
    show_card(
        "Theoretical chance of 1+ false alarm",
        f"{familywise:.2%}",
        "Assuming independent tests and no real differences.",
    )
    st.markdown("**Each test's p-value**")
    chips = "".join(
        f'<span class="p-value {"win" if value < 0.05 else ""}">'
        f'#{index + 1} · {value:.3f}{" · false win" if value < 0.05 else ""}</span>'
        for index, value in enumerate(displayed)
    )
    st.markdown(chips, unsafe_allow_html=True)
    show_callout(
        "<b>Why this happens:</b> Each test has a 5% false-alarm threshold. "
        "Across 20 independent tests, the chance of at least one false alarm "
        "is 1 − 0.95²⁰ = 64.15%.",
        "warn",
    )
    st.caption(
        "P-values shown are illustrative, not simulated. The overall false-alarm "
        "probability is theoretical and assumes independent tests and a true null in each."
    )

st.markdown(
    '<div class="fine-print">Teaching demo · two-sided pooled proportion z-test · '
    '95% unpooled Wald confidence interval. A small p-value is not the probability '
    'that a hypothesis is true.</div>',
    unsafe_allow_html=True,
)
