# /// script
# requires-python = ">=3.12"
# dependencies = ["marimo", "polars", "matplotlib"]
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

    import marimo as mo
    import matplotlib.pyplot as plt
    import polars as pl

    REPO = "https://huggingface.co/datasets/mmmaurer/promptology-results/resolve/main/"
    METRICS = {"accuracy": "balanced accuracy", "f1": "F1"}

    @cache
    def load(name, sep=","):
        raw = urllib.request.urlopen(REPO + name).read()
        return pl.read_csv(io.BytesIO(raw), separator=sep)

    def paper_figure(name):
        return mo.image(str(mo.notebook_location() / "public" / name))

    return METRICS, load, mo, paper_figure, pl, plt


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

    The results load from
    [huggingface.co/datasets/mmmaurer/promptology-results](https://huggingface.co/datasets/mmmaurer/promptology-results).
    """)
    return


@app.cell
def _(mo):
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
    mo.hstack([language, model, metric], justify="start", gap=2)
    return language, metric, model


@app.cell
def _(METRICS, language, load, metric, model, pl, plt):
    df = load("rq1_sbert_results.csv", sep=";").filter(
        pl.col("language") == language.value
    )
    if model.value:
        df = df.filter(pl.col("model") == model.value)

    stats = (
        df.group_by("prompt_condition")
        .agg(
            pl.col(metric.value).mean().alias("mean"),
            pl.col(metric.value).std().alias("sd"),
        )
        .sort("mean")
    )

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.errorbar(
        stats["mean"],
        range(stats.height),
        xerr=stats["sd"],
        fmt="o",
        color="#4C72B0",
        capsize=3,
    )
    ax.set_xlim(right=1.0)
    ax.set_yticks(range(stats.height), stats["prompt_condition"].to_list())
    ax.set_xlabel(f"mean {METRICS[metric.value]}, with one standard deviation")
    ax.set_title(f"Difference analysis per prompt condition ({language.value})")
    fig.tight_layout()
    fig
    return


@app.cell
def _(mo):
    mo.md("""
    ## Style against semantics

    Each classifier separates human arguments from generated ones in one setting. The
    linguistic features cover style. The sentence embeddings cover semantics. A higher
    score means a larger difference between the two sources.

    The file with the linguistic results is about 18 MB. The first run is slow.
    """)
    return


@app.cell
def _(METRICS, language, load, metric, model, pl, plt):
    ling = load("rq1_ling_results.csv", sep=";")
    semantic = load("rq1_sbert_results.csv", sep=";")
    if model.value:
        ling = ling.filter(pl.col("model") == model.value)
        semantic = semantic.filter(pl.col("model") == model.value)
    ling = ling.filter(pl.col("language") == language.value)
    semantic = semantic.filter(pl.col("language") == language.value)

    areas = (
        ling.group_by("feature_area")
        .agg(
            pl.col(metric.value).mean().alias("mean"),
            pl.col(metric.value).std().alias("sd"),
        )
        .sort("mean")
    )

    fig4, ax4 = plt.subplots(figsize=(7, 5), layout="constrained")
    ax4.errorbar(
        areas["mean"],
        range(areas.height),
        xerr=areas["sd"],
        fmt="o",
        color="#DD8452",
        capsize=3,
        label="linguistic features (style)",
    )
    ax4.axvline(
        semantic[metric.value].mean(),
        color="#4C72B0",
        linestyle="--",
        label="sentence embeddings (semantics)",
    )
    ax4.set_xlim(right=1.0)
    ax4.set_yticks(range(areas.height), areas["feature_area"].to_list())
    ax4.set_xlabel(f"mean {METRICS[metric.value]}, with one standard deviation")
    ax4.set_title(f"Difference analysis per feature area ({language.value})")
    ax4.legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=False
    )
    fig4
    return


@app.cell
def _(mo):
    mo.md("""
    ## Effect sizes

    The plot below shows the posterior means and the 94% highest density intervals of
    the regression model on the SBERT classification results. An interval that excludes
    zero marks a credible effect.
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
def _(effects, load, plt):
    summary = load("rq1_sbert_summary.csv").rename({"": "term"})
    rows = summary.filter(
        [effects.value(term) for term in summary["term"]]
    )

    fig2, ax2 = plt.subplots(figsize=(7, 0.35 * rows.height + 1.5))
    lo = rows["mean"] - rows["hdi_3%"]
    hi = rows["hdi_97%"] - rows["mean"]
    ax2.errorbar(
        rows["mean"],
        range(rows.height),
        xerr=[lo, hi],
        fmt="o",
        color="#333333",
        capsize=3,
    )
    ax2.axvline(0, linestyle="--", color="crimson", linewidth=1)
    ax2.set_yticks(range(rows.height), rows["term"])
    ax2.set_xlabel("posterior mean with 94% HDI")
    fig2.tight_layout()
    fig2
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
    with each other. A high score means that the model repeats itself.

    The files for one language are about 10 MB. The first run of a language is slow.
    """)
    return


@app.cell
def _(language, load, plt):
    bert = load(f"bert_score_{language.value}.csv")
    rouge = load(f"rogueL_scores_{language.value}.csv")

    fig3, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=False)
    for ax3, (data, column, title) in zip(
        axes,
        [(bert, "f1", "BERTScore F1"), (rouge, "average_F1", "Rouge-L F1")],
    ):
        models = sorted(data["model"].unique())
        ax3.boxplot(
            [data.filter(data["model"] == m)[column].to_list() for m in models],
            tick_labels=[m.split("-")[0] for m in models],
            showfliers=False,
        )
        ax3.set_title(title)
        ax3.tick_params(axis="x", rotation=30)
    fig3.suptitle(f"Stability of repeated generations ({language.value})")
    fig3.tight_layout()
    fig3
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
