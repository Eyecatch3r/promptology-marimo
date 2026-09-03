"""Schema check for showcase.py. Run: python test_showcase.py"""

import io
import itertools
import urllib.request

import polars as pl

REPO = "https://huggingface.co/datasets/mmmaurer/promptology-results/resolve/main/"


def read(name, sep=","):
    raw = urllib.request.urlopen(REPO + name).read()
    return pl.read_csv(io.BytesIO(raw), separator=sep)


def header(name, sep=";"):
    """Read the first line only. The linguistic results file is 18 MB."""
    request = urllib.request.Request(REPO + name, headers={"Range": "bytes=0-2000"})
    first = urllib.request.urlopen(request).read().split(b"\n")[0]
    return first.decode().strip().split(sep)


def test_switch_labels_match_the_data():
    """The four switches build a label. Every label must exist as a condition."""
    names = ("C", "SD", "Ex", "P&C")
    labels = {
        " + ".join(n for n, on in zip(names, flags) if on) or "Default"
        for flags in itertools.product((0, 1), repeat=4)
    }
    conditions = set(read("rq1_sbert_results.csv", sep=";")["prompt_condition"])
    assert labels == conditions, labels ^ conditions


def test_columns_used_by_the_plots():
    used = {"language", "model", "topic", "stance", "prompt_condition", "accuracy", "f1"}
    assert used <= set(read("rq1_sbert_results.csv", sep=";").columns)
    assert used | {"feature_area"} <= set(header("rq1_ling_results.csv"))


def test_summary_columns_for_the_forest_plot():
    summary = read("rq1_sbert_summary.csv").rename({"": "term"})
    assert {"term", "mean", "hdi_3%", "hdi_97%"} <= set(summary.columns)
    assert "sociodemographics" in set(summary["term"])


def test_prompt_files_in_public():
    """build_prompt_data.py writes these. showcase.py reads them from public/."""
    design = pl.read_csv("public/prompt_design.csv")
    questions = pl.read_csv("public/prompt_questions.csv")
    examples = pl.read_csv("public/prompt_examples.csv")

    assert design.height == 32164, design.height
    assert {"language", "qid", "stance", "condition", "dimension", "group", "chars"} <= set(
        design.columns
    )
    assert questions.height == 120
    assert design["condition"].n_unique() == 16
    assert set(examples["condition"]) == set(design["condition"])


if __name__ == "__main__":
    for name, test in sorted(globals().items()):
        if name.startswith("test_"):
            test()
            print(f"ok  {name}")
