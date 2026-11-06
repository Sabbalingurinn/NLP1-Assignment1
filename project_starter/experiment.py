"""Reusable text classification experiment. See README.md before running."""
import argparse
import hashlib
import itertools
import json
import os
import warnings
from pathlib import Path

import baycomp
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from tokenizers import Tokenizer, models, pre_tokenizers, trainers

SEED = 42
ROPE = 0.01
IMDB_REVISION = "d05f0155defa7991dad75bc68c5ccb6774b1fdc5"
IMDB_ORDERED_SHA256 = "0ed0de756c57a4ac8868e7cc2a9970650e930d94cdb0347716d385c9e1e705be"
REPRESENTATIONS = ("indicator", "counts", "bigrams", "trigrams_hashed")
MODELS = ("logistic", "mlp", "mlp_32_32", "svm")


def prepare_output_dir(path):
    """Never overwrite a previous run, even if it was interrupted."""
    out = Path(path)
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise SystemExit(f"Output already exists and is not empty: {out}. Choose a new --output directory.")
    out.mkdir(parents=True, exist_ok=True)
    return out


def ordered_data_sha256(df):
    """Hash text and label bytes in row order, including unambiguous lengths."""
    digest = hashlib.sha256()
    for row in df[["text", "label"]].itertuples(index=False, name=None):
        for value in row:
            data = value.encode("utf-8")
            digest.update(len(data).to_bytes(8, "big"))
            digest.update(data)
    return digest.hexdigest()


def split_indices(df):
    """Reproduce the project's stratified, individually shuffled 70/15/15 splits."""
    train_idx, other_idx = train_test_split(np.arange(len(df)), test_size=0.30,
                                            stratify=df.label, random_state=SEED)
    val_idx, test_idx = train_test_split(other_idx, test_size=0.5,
                                         stratify=df.label.iloc[other_idx], random_state=SEED)
    rng = np.random.default_rng(SEED)
    return tuple(rng.permutation(indices) for indices in (train_idx, val_idx, test_idx))


def build_features(train_texts, eval_texts, representation, vocab_size):
    # Both tokenizer and vectorizer are fitted ONLY on this fold's training texts.
    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, min_frequency=2,
        special_tokens=["[UNK]"], initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False)
    tokenizer.train_from_iterator(train_texts, trainer=trainer)

    def tokenize(text):
        return tokenizer.encode(text).tokens

    if representation == "trigrams_hashed":
        # Hash counts into fixed-size columns; collisions are possible.
        vec = HashingVectorizer(tokenizer=tokenize, token_pattern=None, lowercase=False,
                                ngram_range=(3, 3), n_features=2**16,
                                alternate_sign=False, norm=None, dtype=np.float32)
        return vec.transform(train_texts), vec.transform(eval_texts)

    vec = CountVectorizer(tokenizer=tokenize, token_pattern=None, lowercase=False,
                          binary=representation == "indicator",
                          ngram_range=(2, 2) if representation == "bigrams" else (1, 1),
                          dtype=np.float64)
    X_train = vec.fit_transform(train_texts)
    X_eval = vec.transform(eval_texts)
    return X_train, X_eval


def predict_model(name, X_train, y_train, X_eval):
    if name == "logistic":
        model = LogisticRegression(max_iter=500, random_state=SEED)
    elif name == "mlp":
        model = MLPClassifier(hidden_layer_sizes=(64,), max_iter=100, random_state=SEED)
    elif name == "mlp_32_32":
        model = MLPClassifier(hidden_layer_sizes=(32, 32), max_iter=100, random_state=SEED)
    elif name == "svm":
        model = LinearSVC(random_state=SEED)
    else:
        raise ValueError(f"Unknown model: {name}")
    model.fit(X_train, y_train)
    return model.predict(X_eval)


def score(y, pred, metric):
    if metric == "accuracy":
        return accuracy_score(y, pred)
    return f1_score(y, pred, average="macro", zero_division=0)


def load_data(args):
    if args.imdb:
        # Cache within the project, so the same downloaded version is reused.
        os.environ.setdefault("HF_HOME", str(Path(__file__).resolve().parent / ".hf_cache"))
        from datasets import load_dataset
        dataset = load_dataset("mteb/imdb", revision=IMDB_REVISION)
        print("Published dataset sizes:", {name: len(dataset[name]) for name in ("train", "test")})
        # Intentionally pool the published splits to implement the requested 70/15/15.
        df = pd.concat([dataset[name].select_columns(["text", "label_text"]).to_pandas()
                        for name in ("train", "test")], ignore_index=True)
        df = df.rename(columns={"label_text": "label"})
        print("Pooling published splits; results are not comparable to the official test benchmark.")
    elif args.demo:
        # Explicitly artificial data: useful only to check that the pipeline runs.
        labels = ["sports", "business", "science"]
        examples = {"sports": "football team scored match", "business": "company market profit shares", "science": "research space experiment discovery"}
        rows = [{"text": f"{examples[label]} report example {i}", "label": label}
                for label in labels for i in range(35)]
        df = pd.DataFrame(rows)
        print("DEMO ONLY: synthetic data, scores have no scientific meaning.")
    else:
        df = pd.read_csv(args.csv)
        if args.text_column not in df or args.label_column not in df:
            raise ValueError("The selected text and label columns must exist in your CSV.")
        df = df[[args.text_column, args.label_column]].rename(columns={args.text_column: "text", args.label_column: "label"})
    if df[["text", "label"]].isna().any().any():
        raise ValueError("Text and labels must not be missing; inspect and clean the dataset first.")
    if (df.text.astype(str).str.strip() == "").any():
        raise ValueError("Empty texts found; inspect and clean the dataset first.")
    df["text"] = df.text.astype(str)
    df["label"] = df.label.astype(str)
    if df.label.nunique() < 2:
        raise ValueError("At least two classes are required.")
    if args.imdb:
        actual = ordered_data_sha256(df)
        if actual != IMDB_ORDERED_SHA256:
            raise ValueError("The IMDb data or row order differs from the saved experiment data; "
                             "do not reuse its split or compare scores without investigating.")
        print("Verified the cached IMDb text and label order against the saved data fingerprint.")
    return df


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--csv", help="CSV file containing one text and label per row")
    source.add_argument("--demo", action="store_true")
    source.add_argument("--imdb", action="store_true", help="Load and pool the published mteb/imdb train and test splits")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--metric", choices=("macro_f1", "accuracy"), default="macro_f1")
    parser.add_argument("--vocab-size", type=int, default=1000)
    parser.add_argument("--output", default="project_starter/results/new_full_comparison")
    parser.add_argument("--final", nargs=2, metavar=("REPRESENTATION", "MODEL"), help="Evaluate this selected system once on the held-out test set")
    args = parser.parse_args()
    if args.vocab_size < 257:
        parser.error("--vocab-size must be at least 257 for the byte alphabet and [UNK].")
    if args.final and (args.final[0] not in REPRESENTATIONS or args.final[1] not in MODELS):
        parser.error("Invalid --final choice; see README for available representations and models.")
    out = prepare_output_dir(args.output)
    df = load_data(args)
    print("Class counts:\n", df.label.value_counts().to_string())
    train_idx, val_idx, test_idx = split_indices(df)
    texts, labels = df.text.to_numpy(), df.label.to_numpy()
    print(f"Split sizes: train={len(train_idx)}, validation={len(val_idx)}, test={len(test_idx)}")
    config = {"source": "mteb/imdb (pooled published train and test)" if args.imdb else (str(args.csv) if args.csv else "synthetic demo"), "seed": SEED, "metric": args.metric,
              "rope": ROPE, "vocab_size": args.vocab_size, "sizes": [len(train_idx), len(val_idx), len(test_idx)],
              "imdb_revision": IMDB_REVISION if args.imdb else None,
              "imdb_ordered_sha256": IMDB_ORDERED_SHA256 if args.imdb else None,
              "final": args.final}
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    if args.final:
        representation, name = args.final
        # Once choices are finalized, refit on train + validation and evaluate test once.
        dev_idx = np.concatenate([train_idx, val_idx])
        X_train, X_test = build_features(texts[dev_idx], texts[test_idx], representation, args.vocab_size)
        pred = predict_model(name, X_train, labels[dev_idx], X_test)
        report = classification_report(labels[test_idx], pred, zero_division=0)
        print(f"Held-out test: {representation}/{name}; {args.metric}={score(labels[test_idx], pred, args.metric):.4f}\n{report}")
        (out / "test_report.txt").write_text(f"{representation}/{name}; {args.metric}={score(labels[test_idx], pred, args.metric):.4f}\n{report}")
        return

    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    fold_splits = list(splitter.split(texts[train_idx], labels[train_idx]))
    records = []
    for rep in REPRESENTATIONS:
        for fold, (fit_pos, eval_pos) in enumerate(fold_splits, 1):
            fit_idx, eval_idx = train_idx[fit_pos], train_idx[eval_pos]
            X_fit, X_eval = build_features(texts[fit_idx], texts[eval_idx], rep, args.vocab_size)
            for name in MODELS:
                print(f"CV: {rep} / {name} / fold {fold}", flush=True)
                pred = predict_model(name, X_fit, labels[fit_idx], X_eval)
                records.append({"representation": rep, "model": name, "fold": fold, "score": score(labels[eval_idx], pred, args.metric)})
    scores = pd.DataFrame(records)
    scores.to_csv(out / "cv_scores.csv", index=False)
    print("\nCross-validation:\n", scores.groupby(["representation", "model"]).score.agg(["mean", "std"]).sort_values("mean", ascending=False).to_string())

    validation = []
    for rep in REPRESENTATIONS:
        X_fit, X_val = build_features(texts[train_idx], texts[val_idx], rep, args.vocab_size)
        for name in MODELS:
            print(f"Validation: {rep} / {name}", flush=True)
            pred = predict_model(name, X_fit, labels[train_idx], X_val)
            validation.append({"representation": rep, "model": name, "score": score(labels[val_idx], pred, args.metric)})
    val_results = pd.DataFrame(validation).sort_values("score", ascending=False)
    val_results.to_csv(out / "validation_scores.csv", index=False)
    print("\nValidation:\n", val_results.to_string(index=False))

    # Paired scores from identical folds; baycomp returns P(A>B), P(ROPE), P(B>A).
    systems = list(scores.groupby(["representation", "model"], sort=True))
    comparisons = []
    for (a, ar), (b, br) in itertools.combinations(systems, 2):
        x = ar.sort_values("fold").score.to_numpy()
        y = br.sort_values("fold").score.to_numpy()
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", RuntimeWarning)
                probs = baycomp.CorrelatedTTest(x, y, rope=ROPE, runs=1).probs()
            if not np.all(np.isfinite(probs)):
                raise ValueError("non-finite posterior")
            comparisons.append({"A": "/".join(a), "B": "/".join(b), "P_A_better": probs[0], "P_within_ROPE": probs[1], "P_B_better": probs[2], "note": ""})
        except (ValueError, RuntimeWarning, ZeroDivisionError) as exc:
            comparisons.append({"A": "/".join(a), "B": "/".join(b), "P_A_better": np.nan, "P_within_ROPE": np.nan, "P_B_better": np.nan, "note": str(exc)})
    pd.DataFrame(comparisons).to_csv(out / "bayesian_comparisons.csv", index=False)
    print(f"\nResults saved in {out}. Review CV, validation, and Bayesian comparisons before choosing a final system.")


if __name__ == "__main__":
    main()
