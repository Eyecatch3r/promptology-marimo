import polars as pl
from sklearn import svm
from sklearn.metrics import (
    precision_score, recall_score, f1_score, balanced_accuracy_score
)
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
import numpy as np
import os
import argparse
import itertools

def run_experiments(path_to_file):
    # create results directory if it doesn't exist
    if not os.path.exists("results"):
        os.makedirs("results")

    # load the data
    df = pl.read_csv(path_to_file)

    # add label column
    df = df.with_columns(
        pl.when(pl.col("model") == "Human")
        .then(pl.lit(1)).otherwise(pl.lit(0)).alias("label")
    )

    feats = df.columns[2:164]

    emotion = [
        col for col in feats if "intensity" in col
        or "sentiment" in col
        or "valence" in col
        or "arousal" in col
        or "dominance" in col
    ]
    semantics = [col for col in feats if "synsets" in col]
    readability = [
        'n_syllables',
        'n_monosyllables',
        'n_polysyllables',
        'flesch_reading_ease',
        'flesch_kincaid_grade',
        'gunning_fog',
        'ari',
        'smog',
        'cli',
        'lix',
        'rix'
    ]
    # only concreteness is present in all three languages
    psycholinguistics = [col for col in feats if "concreteness" in col]
    information = ['compressibility', 'entropy']
    morphological = [col for col in feats if "VERB" in col
                     or "NOUN" in col or "ADJ" in col or "ADV" in col
                     or "PRON" in col or "ADP" in col or "AUX" in col
                     or "CCONJ" in col or "DET" in col or "NUM" in col
                     or "PROPN" in col or "PUNCT" in col or "SCONJ" in col
                     or "VERB" in col or "X" in col]
    surface = [
        'raw_sequence_length',
        'n_tokens',
        'n_sentences',
        'tokens_per_sentence',
        'n_characters',
        'avg_word_length',
        'n_types',
        'n_long_words',
        'n_lemmas'
    ]
    pos = [
        'n_adj',
        'n_adp',
        'n_adv',
        'n_aux',
        'n_cconj',
        'n_det',
        'n_noun',
        'n_num',
        'n_pron',
        'n_propn',
        'n_punct',
        'n_sconj',
        'n_verb',
        'n_x'
    ]
    lexical_richness = [
        'lemma_token_ratio',
        'ttr',
        'rttr',
        'cttr',
        'herdan_c',
        'summer_index',
        'dugast_u',
        'maas_index',
        'n_lexical_tokens',
        'lexical_density',
        'n_hapax_legomena',
        'n_global_token_hapax_legomena',
        'n_global_lemma_hapax_legomena',
        'n_hapax_dislegomena',
        'n_global_token_hapax_dislegomena',
        'n_global_lemma_hapax_dislegomena',
        'sichel_s',
        'global_sichel_s',
        'giroud_index',
        'mtld',
        'mattr',
        'msttr',
        'yule_k',
        'simpsons_d',
        'herdan_v',
        'pos_variability'
    ]
    named_entities = [
        'n_entities',
        'n_org',
        'n_loc'
    ]
    dependencies = [
        'tree_width',
        'tree_depth',
        'tree_branching',
        'n_noun_chunks',
        'n_dependency_cc',
        'n_dependency_dep',
        'n_dependency_punct'   
    ]

    feats.remove("hdd")
    df = df.drop("hdd")

    areas = [named_entities, emotion, semantics, readability,
             psycholinguistics, information, surface, pos,
             lexical_richness, morphological, dependencies, feats]
    area_names = ["named_entities", "emotion", "semantics", "readability",
                  "psycholinguistics", "information", "surface", "pos",
                  "lexical_richness", "morphological", "dependencies", 
                  "all_features"]

    res_file = "results/rq1_results.csv"
    with open(res_file, "w") as f:
        f.write("fold;"
                "n_train_positive;"
                "n_train_negative;"
                "n_test_positive;"
                "n_test_negative;"
                "model;"
                "prompt_condition;"
                "language;"
                "topic;"
                "stance;"
                "feature_area;"
                "accuracy;"
                "precision;"
                "recall;"
                "f1;"
                "tn;"
                "fp;"
                "fn;"
                "tp\n"
                )

    # the following are hard-coded to ensure deterministic results across 
    # different runs/feature sets.
    conditions = [
        'Default',
        'SD',
        'C',
        'Ex',
        'P&C',
        'C + Ex',
        'C + P&C',
        'SD + Ex',
        'SD + P&C',
        'Ex + P&C',
        'C + SD',
        'C + SD + P&C',
        'C + Ex + P&C',
        'C + SD + Ex',
        'SD + Ex + P&C',
        'C + SD + Ex + P&C'
    ]

    models = [
        'gpt-4.1-mini',
        'llama_4_scout',
        'llama_3.1-8b-instruct',
        'occiglot-7b-eu5-instruct'
 ]
    stances = ["FAVOR", "AGAINST"]

    topics = [
        'Democracy, Media & Digitization',
        'Energy & transport',
        'Health',
        'Nature conservation',
        'Values',
        'Security & military',
        'Welfare state & family',
        'Foreign trade & foreign policy',
        'Economy & labour',
        'Federal budget',
        'Education',
        'Society & ethics',
        'Immigration & integration'
    ]

    languages = ["de", "fr", "it"]
    
    combinations = list(itertools.product(
         languages, topics, conditions, models, stances))
        
    for area, area_name in zip(areas, area_names):
        for lang, topic, condition, model, stance in combinations:
            subset = df.filter(
                (pl.col("language") == lang),
                (pl.col("topic") == topic),
                (pl.col("condition").is_in([condition, "Human"])),
                (pl.col("model").is_in([model, "Human"])),
                (pl.col("stance") == stance)
            )

            # skip if there are fewer than 100 samples to avoid running
            # experiments on very small datasets which would lead to
            # unreliable results
            if subset.shape[0] < 100:
                continue

            # label and features
            X = subset.select(pl.col(area)).to_numpy()
            y = subset.select(pl.col("label")).to_numpy()

            # argument_id for error analysis later on
            argument_ids = subset.select(pl.col("argument_id")).to_numpy()

            kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            
            for fold, (train_index, test_index) in enumerate(kf.split(X, y)):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]
                ids_test = argument_ids[test_index]

                n_train_positive = sum(y_train)[0]
                n_train_negative = len(y_train) - n_train_positive
                n_test_positive = sum(y_test)[0]
                n_test_negative = len(y_test) - n_test_positive

                if n_train_positive == 0 or n_train_negative == 0 or  \
                    n_test_positive == 0 or n_test_negative == 0:
                    # skip this fold if there are no positive or negative 
                    # samples in either train or test set
                    continue
                clf = svm.SVC(kernel="linear", random_state=42,
                                  class_weight="balanced",
                                  probability=True)
                clf.fit(X_train, y_train.ravel())
                y_pred = clf.predict(X_test)
                raw_scores = clf.decision_function(X_test)
                probs = clf.predict_proba(X_test)
                pos_probs = probs[:, 1]  # probability of being classified as "Human"
                neg_probs = probs[:, 0]  # probability of being classified as "Model"

                # save raw predictions for error analysis later on
                predictions_df = pl.DataFrame({
                    "argument_id": ids_test.ravel(),
                    "true_label": y_test.ravel(),
                    "predicted_label": y_pred.ravel(),
                    "raw_prediction_score": raw_scores.ravel(),
                    "prob_human": pos_probs.ravel(),
                    "prob_model": neg_probs.ravel(),
                    "fold": fold,
                    "model": model,
                    "condition": condition,
                    "language": lang,
                    "topic": topic,
                    "stance": stance,
                    "feature_area": area_name
                })

                tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

                acc = balanced_accuracy_score(y_test, y_pred)
                prec = precision_score(y_test, y_pred)
                rec = recall_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred, average="weighted")

                with open(res_file, "a") as f:
                    f.write(f"{fold};"
                            f"{n_train_positive};"
                            f"{n_train_negative};"
                            f"{n_test_positive};"
                            f"{n_test_negative};"
                            f"{model};"
                            f"{condition};"
                            f"{lang};"
                            f"{topic};"
                            f"{stance};"
                            f"{area_name};"
                            f"{acc:.4f};"
                            f"{prec:.4f};"
                            f"{rec:.4f};"
                            f"{f1:.4f};"
                            f"{tn};"
                            f"{fp};"
                            f"{fn};"
                            f"{tp}\n")
                # save predictions for error analysis later on
                try:
                    preds = pl.read_csv("results/rq1_ling_predictions.csv",
                                        schema_overrides={
                                            "argument_id": pl.Utf8,
                                            "true_label": pl.Int32,
                                            "predicted_label": pl.Int32,
                                            "raw_prediction_score": pl.Float64,
                                            "prob_human": pl.Float64,
                                            "prob_model": pl.Float64,
                                            "fold": pl.Int32,
                                            "model": pl.Utf8,
                                            "condition": pl.Utf8,
                                            "language": pl.Utf8,
                                            "topic": pl.Utf8,
                                            "stance": pl.Utf8,
                                            "feature_area": pl.Utf8
                                        })
                except FileNotFoundError:
                    preds = pl.DataFrame()
                preds = pl.concat([
                    preds,
                    predictions_df
                ])
                preds.write_csv("results/rq1_ling_predictions.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run experiments for RQ1")
    parser.add_argument("--path_to_file", type=str, required=True,
                        help="Path to the CSV file containing the features "
                        "and labels")
    args = parser.parse_args()

    run_experiments(args.path_to_file)

