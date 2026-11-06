"""Optionally evaluate one of the two paired-CV candidates on a held-out split."""

import argparse
import json
import platform
import time
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

import experiment as ex
import top2_cross_validation as top2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', required=True, choices=[item[0] for item in top2.SETTINGS])
    parser.add_argument('--stage', required=True, choices=('validation', 'test'),
                        help='Validation fits on training only; test fits on training + validation.')
    parser.add_argument('--demo', action='store_true', help='Use synthetic data for a quick code check.')
    parser.add_argument('--output', help='New or empty output directory; existing results are never overwritten.')
    args = parser.parse_args()

    prefix = 'demo_' if args.demo else ''
    default_output = f'project_starter/results/{prefix}{args.candidate}_{args.stage}'
    out = ex.prepare_output_dir(args.output or default_output)

    source = argparse.Namespace(imdb=not args.demo, demo=args.demo, csv=None,
                                text_column='text', label_column='label')
    df = ex.load_data(source)
    texts, labels = df.text.to_numpy(), df.label.to_numpy()
    train_idx, val_idx, test_idx = ex.split_indices(df)
    if args.stage == 'validation':
        fit_idx, eval_idx = train_idx, val_idx
    else:
        fit_idx, eval_idx = np.concatenate([train_idx, val_idx]), test_idx

    _, min_df, ngram = next(item for item in top2.SETTINGS if item[0] == args.candidate)
    config = {
        'source': 'synthetic demo' if args.demo else 'mteb/imdb pooled published splits',
        'imdb_revision': None if args.demo else ex.IMDB_REVISION,
        'ordered_text_label_sha256': ex.ordered_data_sha256(df),
        'candidate': args.candidate,
        'stage': args.stage,
        'seed': ex.SEED,
        'metric': 'macro_f1',
        'split_sizes': {'train': len(train_idx), 'validation': len(val_idx), 'test': len(test_idx)},
        'fit_size': len(fit_idx),
        'evaluation_size': len(eval_idx),
        'fit_partition': 'train' if args.stage == 'validation' else 'train + validation',
        'bpe_vocab_limit': 2000,
        'bpe_min_pair_frequency': 2,
        'min_df_documents': min_df,
        'ngram_range': list(ngram),
        'max_features': 100000,
        'dtype': 'float32',
        'mlp_hidden_layer_sizes': [64],
        'mlp_activation': 'relu',
        'mlp_solver': 'adam',
        'mlp_max_iter': 100,
        'python': platform.python_version(),
    }
    (out / 'config.json').write_text(json.dumps(config, indent=2) + '\n')

    print(f"Fitting {args.candidate} on {len(fit_idx):,} records for {args.stage} evaluation.", flush=True)
    started = time.monotonic()
    tokenizer = top2.build_tokenizer(texts[fit_idx])
    vectorizer = top2.build_vectorizer(tokenizer, min_df, ngram)
    x_fit = vectorizer.fit_transform(texts[fit_idx])
    x_eval = vectorizer.transform(texts[eval_idx])
    feature_seconds = time.monotonic() - started

    model = top2.build_model()
    started = time.monotonic()
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter('always')
        model.fit(x_fit, labels[fit_idx])
    predictions = model.predict(x_eval)
    fit_seconds = time.monotonic() - started
    class_names = sorted(np.unique(labels).tolist())
    report = classification_report(labels[eval_idx], predictions, zero_division=0)
    metrics = {
        'candidate': args.candidate,
        'stage': args.stage,
        'macro_f1': f1_score(labels[eval_idx], predictions, average='macro', zero_division=0),
        'accuracy': accuracy_score(labels[eval_idx], predictions),
        'classification_report': classification_report(labels[eval_idx], predictions,
                                                       output_dict=True, zero_division=0),
        'confusion_matrix_labels': class_names,
        'confusion_matrix': confusion_matrix(labels[eval_idx], predictions, labels=class_names).tolist(),
        'features': x_fit.shape[1],
        'epochs': int(model.n_iter_),
        'convergence_warning': any(issubclass(w.category, ConvergenceWarning) for w in captured),
        'warnings': [str(w.message) for w in captured],
        'feature_seconds': round(feature_seconds, 2),
        'fit_and_predict_seconds': round(fit_seconds, 2),
    }
    (out / 'metrics.json').write_text(json.dumps(metrics, indent=2) + '\n')
    (out / 'report.txt').write_text(
        f"{args.candidate} / {args.stage}; macro F1 = {metrics['macro_f1']:.4f}\n\n{report}\n"
    )
    print(f"\n{args.stage.capitalize()} macro F1: {metrics['macro_f1']:.4f}\n{report}")
    print(f'Results saved in {out}.')


if __name__ == '__main__':
    main()
