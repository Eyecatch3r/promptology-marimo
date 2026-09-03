"""Build the small prompt files that showcase.py reads from public/.

The three files in prompts/ are 84 MB, because every row repeats the question,
the explanation and the pro and contra texts. This script normalizes them:

    public/prompt_design.csv     one row per prompt, no long text
    public/prompt_questions.csv  one row per question
    public/prompt_examples.csv   one real prompt per condition and stance

Run: python build_prompt_data.py
"""

import polars as pl

LANGUAGES = ("de", "fr", "it")
ELEMENTS = (("context", "C"), ("sd", "SD"), ("ex", "Ex"), ("pc", "P&C"))
OUT = "public"


def condition_label(row):
    """Same label as the prompt_condition column in the result files."""
    return " + ".join(name for flag, name in ELEMENTS if row[flag]) or "Default"


def read(language):
    frame = pl.read_csv(f"prompts/prompts_{language}.csv")
    return frame.with_columns(
        language=pl.lit(language),
        context=pl.col("context").cast(pl.Boolean),
        sd=pl.col("sociodemographic_info").is_not_null(),
        ex=pl.col("info").is_not_null(),
        pc=pl.col("pro").is_not_null(),
        chars=pl.col("prompt").str.len_chars(),
        qid=pl.col("question").rank("dense").cast(pl.Int32),
    )


def main():
    frames = [read(language) for language in LANGUAGES]
    joined = pl.concat(frames)

    design = joined.select(
        "id",
        "language",
        "qid",
        "stance",
        "context",
        "sd",
        "ex",
        "pc",
        "chars",
        dimension=pl.col("sociodemographic_info").fill_null("none"),
        group=pl.col("sociodemographic_group").fill_null("none"),
    ).with_columns(
        condition=pl.struct("context", "sd", "ex", "pc").map_elements(
            condition_label, return_dtype=pl.String
        )
    )

    questions = joined.select("language", "qid", "question", "info", "pro", "contra").unique(
        subset=["language", "qid"], maintain_order=True
    )

    examples = (
        joined.sort("id")
        .with_columns(
            condition=pl.struct("context", "sd", "ex", "pc").map_elements(
                condition_label, return_dtype=pl.String
            )
        )
        .group_by("language", "condition", "stance", maintain_order=True)
        .first()
        .select("language", "condition", "stance", "qid", "prompt",
                group=pl.col("sociodemographic_group").fill_null("none"))
    )

    for name, frame in (
        ("prompt_design", design),
        ("prompt_questions", questions),
        ("prompt_examples", examples),
    ):
        path = f"{OUT}/{name}.csv"
        frame.write_csv(path)
        print(f"{path}: {frame.height} rows")

    # The labels must match the result files, or the switches filter nothing.
    from_results = set(
        pl.read_csv(
            "https://huggingface.co/datasets/mmmaurer/promptology-results/"
            "resolve/main/rq1_sbert_results.csv",
            separator=";",
        )["prompt_condition"]
    )
    assert set(design["condition"]) == from_results, "condition labels drifted"
    print(f"{len(from_results)} condition labels match the result files")


if __name__ == "__main__":
    main()
