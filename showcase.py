# /// script
# requires-python = ">=3.12"
# dependencies = ["marimo", "polars", "altair"]
# ///
"""Interactive showcase of the Promptology results. Run: marimo edit showcase.py"""

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import io
    import urllib.request
    from functools import cache

    import altair as alt
    import marimo as mo
    import polars as pl

    REPO = "https://huggingface.co/datasets/mmmaurer/promptology-results/resolve/main/"
    METRICS = {"accuracy": "balanced accuracy", "f1": "F1"}

    @cache
    def load(name, sep=","):
        raw = urllib.request.urlopen(REPO + name).read()
        return pl.read_csv(io.BytesIO(raw), separator=sep)

    def paper_figure(name):
        return mo.image(str(mo.notebook_location() / "public" / name))

    def subset(frame, language, model=None, topic=None, stance=None):
        frame = frame.filter(pl.col("language") == language)
        for column, value in (("model", model), ("topic", topic), ("stance", stance)):
            if value:
                frame = frame.filter(pl.col(column) == value)
        return frame

    def spread(frame):
        """Mean with one standard deviation, bounded by the maximum score."""
        return frame.with_columns(
            low=pl.col("mean") - pl.col("sd"),
            high=pl.min_horizontal(pl.col("mean") + pl.col("sd"), 1.0),
        )

    def interval_chart(frame, label, x_title, tooltip, point_color, zero=False):
        """One row per category: an interval with a point. Hover reads the numbers."""
        base = alt.Chart(frame).encode(
            y=alt.Y(f"{label}:N", title=None, sort=frame[label].to_list()),
            tooltip=tooltip,
        )
        return (
            base.mark_rule(strokeWidth=1.5, color="#9AA7B4").encode(
                x=alt.X("low:Q", title=x_title, scale=alt.Scale(zero=zero)),
                x2="high:Q",
            )
            + base.mark_circle(size=90, opacity=1).encode(
                x="mean:Q", color=point_color
            )
        ).properties(width=520, height=24 * frame.height + 20)

    return METRICS, alt, interval_chart, load, mo, paper_figure, pl, spread, subset


@app.cell
def _(mo):
    mo.md("""
    # Promptology

    Large-scale systematic analysis of prompt elements and contextual factors in
    argument generation (EMNLP 2026).

    The study crosses four prompt elements in a full factorial design:

    - **C** context of the debate
    - **SD** sociodemographics of the target audience
    - **Ex** explanation of the argument
    - **P&C** premise and conclusion examples

    Four models generate arguments in German, French and Italian. Two analyses follow.
    The difference analysis asks if a classifier can separate the output of two prompt
    conditions. The stability analysis measures how much the output changes between
    repeated generations.

    Hover any point to read its numbers. Drag across the first plot to filter the table
    below it. The results load from
    [huggingface.co/datasets/mmmaurer/promptology-results](https://huggingface.co/datasets/mmmaurer/promptology-results).
    """)
    return


@app.cell
def _(load, mo):
    topics = sorted(load("rq1_sbert_results.csv", sep=";")["topic"].unique())

    language = mo.ui.dropdown(
        {"German": "de", "French": "fr", "Italian": "it"}, value="German"
    )
    model = mo.ui.dropdown(
        {
            "all models": None,
            "GPT-4.1-mini": "gpt-4.1-mini",
            "LLaMA 3.1 8B": "llama_3.1-8b-instruct",
            "LLaMA 4 Scout": "llama_4_scout",
            "Occiglot 7B": "occiglot-7b-eu5-instruct",
        },
        value="all models",
    )
    metric = mo.ui.dropdown(
        {"Balanced accuracy": "accuracy", "F1": "f1"}, value="Balanced accuracy"
    )
    topic = mo.ui.dropdown(
        {"all topics": None} | {t: t for t in topics}, value="all topics"
    )
    stance = mo.ui.dropdown(
        {"both stances": None, "in favor": "FAVOR", "against": "AGAINST"},
        value="both stances",
    )
    mo.vstack(
        [
            mo.hstack([language, model, metric], justify="start", gap=2),
            mo.hstack([topic, stance], justify="start", gap=2),
        ]
    )
    return language, metric, model, stance, topic


@app.cell
def _(mo):
    ELEMENT_NAMES = ("C", "SD", "Ex", "P&C")
    # marimo tracks a collection of elements only through mo.ui.array.
    elements = mo.ui.array(
        [mo.ui.switch(label=name, value=name == "SD") for name in ELEMENT_NAMES]
    )
    mo.hstack(
        [mo.md("**Prompt elements:**"), *[e for e in elements]],
        justify="start",
        gap=1.5,
    )
    return ELEMENT_NAMES, elements


@app.cell
def _(ELEMENT_NAMES, elements, mo):
    condition = (
        " + ".join(n for n, on in zip(ELEMENT_NAMES, elements.value) if on) or "Default"
    )
    mo.md(f"Selected condition: `{condition}`. The red point marks it in the plot.")
    return (condition,)


@app.cell
def _(
    METRICS,
    alt,
    condition,
    interval_chart,
    language,
    load,
    metric,
    mo,
    model,
    pl,
    spread,
    stance,
    subset,
    topic,
):
    df = subset(
        load("rq1_sbert_results.csv", sep=";"),
        language.value,
        model.value,
        topic.value,
        stance.value,
    )
    mo.stop(
        df.is_empty(),
        mo.md("**No classifiers for this combination.** Widen the filters."),
    )

    stats = spread(
        df.group_by("prompt_condition")
        .agg(
            pl.col(metric.value).mean().alias("mean"),
            pl.col(metric.value).std().alias("sd"),
        )
        .sort("mean", descending=True)
    )

    conditions = mo.ui.altair_chart(
        interval_chart(
            stats,
            "prompt_condition",
            f"mean {METRICS[metric.value]}, with one standard deviation",
            [
                alt.Tooltip("prompt_condition:N", title="condition"),
                alt.Tooltip("mean:Q", format=".3f", title="mean"),
                alt.Tooltip("sd:Q", format=".3f", title="std. dev."),
            ],
            alt.condition(
                alt.datum.prompt_condition == condition,
                alt.value("crimson"),
                alt.value("#4C72B0"),
            ),
        ).properties(
            title=f"Difference analysis per prompt condition ({language.value})"
        )
    )
    conditions
    return (conditions,)


@app.cell
def _(
    condition,
    conditions,
    language,
    load,
    metric,
    mo,
    model,
    pl,
    stance,
    subset,
    topic,
):
    def per_condition(name, file, area=None):
        frame = subset(
            load(file, sep=";"), language.value, model.value, topic.value, stance.value
        )
        if area:
            frame = frame.filter(pl.col("feature_area") == area)
        return frame.group_by("prompt_condition").agg(
            pl.col(metric.value).mean().round(3).alias(name)
        )

    scores = (
        per_condition("semantics", "rq1_sbert_results.csv")
        .join(
            per_condition("style", "rq1_ling_results.csv", "all_features"),
            on="prompt_condition",
            how="full",
            coalesce=True,
        )
        .sort("semantics", descending=True)
    )

    brushed = (
        list(conditions.value["prompt_condition"])
        if conditions.value is not None and len(conditions.value)
        else []
    )
    shown = (
        scores.filter(pl.col("prompt_condition").is_in(brushed)) if brushed else scores
    )
    picked = scores.filter(pl.col("prompt_condition") == condition)

    def score_of(column):
        if not picked.height or picked[column][0] is None:
            return "-"
        return f"{picked[column][0]:.3f}"

    mo.vstack(
        [
            mo.hstack(
                [
                    mo.stat(
                        score_of("semantics"),
                        label="semantics",
                        caption=f"{condition}, sentence embeddings",
                    ),
                    mo.stat(
                        score_of("style"),
                        label="style",
                        caption=f"{condition}, linguistic features",
                    ),
                ],
                justify="start",
                gap=2,
            ),
            mo.md(
                f"{shown.height} of {scores.height} conditions"
                + (", selected in the plot" if brushed else "")
            ),
            mo.ui.table(shown, selection=None, page_size=8),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ## Style against semantics

    Each classifier separates human arguments from generated ones in one setting. The
    linguistic features cover style. The sentence embeddings cover semantics. A higher
    score means a larger difference between the two sources. The dashed line marks the
    sentence embeddings.

    The file with the linguistic results is about 18 MB. The first run is slow.
    """)
    return


@app.cell
def _(
    METRICS,
    alt,
    interval_chart,
    language,
    load,
    metric,
    mo,
    model,
    pl,
    spread,
    stance,
    subset,
    topic,
):
    ling = subset(
        load("rq1_ling_results.csv", sep=";"),
        language.value,
        model.value,
        topic.value,
        stance.value,
    )
    semantic = subset(
        load("rq1_sbert_results.csv", sep=";"),
        language.value,
        model.value,
        topic.value,
        stance.value,
    )
    mo.stop(ling.is_empty(), mo.md("**No classifiers for this combination.**"))

    areas = spread(
        ling.group_by("feature_area")
        .agg(
            pl.col(metric.value).mean().alias("mean"),
            pl.col(metric.value).std().alias("sd"),
        )
        .sort("mean", descending=True)
    )

    (
        interval_chart(
            areas,
            "feature_area",
            f"mean {METRICS[metric.value]}, with one standard deviation",
            [
                alt.Tooltip("feature_area:N", title="feature area"),
                alt.Tooltip("mean:Q", format=".3f", title="mean"),
                alt.Tooltip("sd:Q", format=".3f", title="std. dev."),
            ],
            alt.value("#DD8452"),
        )
        + alt.Chart(pl.DataFrame({"mean": [semantic[metric.value].mean()]}))
        .mark_rule(strokeDash=[5, 4], strokeWidth=2, color="#4C72B0")
        .encode(
            x="mean:Q",
            tooltip=[alt.Tooltip("mean:Q", format=".3f", title="semantics")],
        )
    ).properties(title=f"Style against semantics ({language.value})")
    return


@app.cell
def _(mo):
    mo.md("""
    ## Effect sizes

    The plot below shows the posterior means and the 94% highest density intervals of
    the regression model on the SBERT classification results. A dark point marks an
    interval that excludes zero.
    """)
    return


@app.cell
def _(mo):
    ELEMENTS = ("context", "examples", "explanation", "sociodemographics")
    GROUPS = {
        "Prompt elements": lambda t: t in ELEMENTS,
        "Prompt element interactions": lambda t: ":" in t and "[" not in t,
        "Model, language, stance": lambda t: t.startswith(
            ("model[", "language[", "stance[")
        ),
        "Topics": lambda t: t.startswith("topic["),
    }
    effects = mo.ui.dropdown(GROUPS, value="Prompt elements")
    effects
    return (effects,)


@app.cell
def _(alt, effects, interval_chart, load, pl):
    summary = load("rq1_sbert_summary.csv").rename({"": "term"})
    rows = summary.filter([effects.value(term) for term in summary["term"]]).rename(
        {"hdi_3%": "low", "hdi_97%": "high"}
    )

    (
        interval_chart(
            rows,
            "term",
            "posterior mean with 94% HDI",
            [
                alt.Tooltip("term:N", title="term"),
                alt.Tooltip("mean:Q", format=".3f", title="mean"),
                alt.Tooltip("low:Q", format=".3f", title="HDI 3%"),
                alt.Tooltip("high:Q", format=".3f", title="HDI 97%"),
            ],
            alt.condition(
                alt.datum.low * alt.datum.high > 0,
                alt.value("#333333"),
                alt.value("#B0B0B0"),
            ),
            zero=True,
        )
        + alt.Chart(pl.DataFrame({"zero": [0.0]}))
        .mark_rule(strokeDash=[4, 4], color="crimson")
        .encode(x="zero:Q")
    )
    return


@app.cell
def _(mo, paper_figure):
    mo.vstack(
        [
            mo.md(
                """
                The plot above covers the semantic feature set. The published figure below
                shows both feature sets at the 95% HDI. `elfen` stands for the linguistic
                features, SBERT for the sentence embeddings.
                """
            ),
            paper_figure("figure3_effects_both.svg"),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ## Stability

    Each prompt runs several times. The scores below compare the outputs of one prompt
    with each other. A high score means that the model repeats itself. The bar covers
    the quartiles, the line covers the 5th to the 95th percentile.

    The files for one language are about 10 MB. The first run of a language is slow.
    """)
    return


@app.cell
def _(alt, language, load, pl):
    QUANTILES = {"p5": 0.05, "q1": 0.25, "median": 0.5, "q3": 0.75, "p95": 0.95}

    def quantiles(frame, column):
        return (
            frame.group_by("model")
            .agg(
                [
                    pl.col(column).quantile(q).alias(name)
                    for name, q in QUANTILES.items()
                ]
            )
            .sort("model")
        )

    def box_chart(frame, title):
        base = alt.Chart(frame).encode(
            y=alt.Y("model:N", title=None),
            tooltip=[
                alt.Tooltip("model:N", title="model"),
                alt.Tooltip("median:Q", format=".3f", title="median"),
                alt.Tooltip("q1:Q", format=".3f", title="25%"),
                alt.Tooltip("q3:Q", format=".3f", title="75%"),
                alt.Tooltip("p5:Q", format=".3f", title="5%"),
                alt.Tooltip("p95:Q", format=".3f", title="95%"),
            ],
        )
        return (
            base.mark_rule(strokeWidth=1.5, color="#666666").encode(
                x=alt.X("p5:Q", title="F1", scale=alt.Scale(zero=False)), x2="p95:Q"
            )
            + base.mark_bar(size=18, color="#4C72B0").encode(x="q1:Q", x2="q3:Q")
            + base.mark_tick(size=18, thickness=2, color="white").encode(x="median:Q")
        ).properties(width=250, height=32 * frame.height + 20, title=title)

    (
        box_chart(
            quantiles(load(f"bert_score_{language.value}.csv"), "f1"), "BERTScore F1"
        )
        | box_chart(
            quantiles(load(f"rogueL_scores_{language.value}.csv"), "average_F1"),
            "Rouge-L F1",
        )
    )
    return


@app.cell
def _(mo, paper_figure):
    mo.vstack(
        [
            mo.md(
                """
                ## Effects on stability

                The published figure below shows the regression results for both stability
                metrics. Model, language and topic move the scores more than the prompt
                elements do.
                """
            ),
            paper_figure("figure6_stability_effects.svg"),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
