"""Paired five-fold comparison of the two best one-fold screening configurations."""
import argparse
import json
import os
import platform
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from tokenizers import Tokenizer, models, pre_tokenizers, trainers
import baycomp
import experiment as ex

SETTINGS = (('bpe2000_bigrams', 1, (2, 2)), ('bpe2000_rare_unigrams_bigrams', 2, (1, 2)))


def build_tokenizer(fit_text):
    tok = Tokenizer(models.BPE(unk_token='[UNK]'))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
    trainer = trainers.BpeTrainer(vocab_size=2000, min_frequency=2, special_tokens=['[UNK]'],
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False)
    tok.train_from_iterator(fit_text, trainer=trainer)
    return tok


def build_vectorizer(tok, min_df, ngram):
    return CountVectorizer(tokenizer=lambda text: tok.encode(text).tokens, token_pattern=None,
                           lowercase=False, ngram_range=ngram, min_df=min_df,
                           max_features=100000, dtype=np.float32)


def build_model():
    return MLPClassifier(hidden_layer_sizes=(64,), max_iter=100, random_state=ex.SEED)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--demo', action='store_true')
    parser.add_argument('--output', default='project_starter/results/new_top2_cv')
    args = parser.parse_args()
    out = ex.prepare_output_dir(args.output)
    source = argparse.Namespace(imdb=not args.demo, demo=args.demo, csv=None, text_column='text', label_column='label')
    df = ex.load_data(source)
    texts, labels = df.text.to_numpy(), df.label.to_numpy()
    train_idx, val_idx, test_idx = ex.split_indices(df)
    config = {'source': 'synthetic demo' if args.demo else 'mteb/imdb pooled published splits',
              'seed': ex.SEED, 'folds': 5, 'training_size': len(train_idx), 'validation_size': len(val_idx),
              'test_size': len(test_idx), 'metric': 'macro_f1', 'rope': ex.ROPE,
              'bpe_vocab_limit': 2000, 'bpe_min_pair_frequency': 2, 'min_df': 'number of training documents',
              'max_features': 100000, 'dtype': 'float32', 'mlp_hidden_layer_sizes': [64],
              'mlp_activation': 'relu', 'mlp_solver': 'adam', 'mlp_max_iter': 100,
              'settings': [{'name': n, 'min_df': d, 'ngram_range': list(g)} for n,d,g in SETTINGS],
              'split_policy': 'both tokenizer and feature vocabulary fitted only on each fold training data',
              'untouched': 'separate 15% validation and 15% test; neither used in this run',
              'python': platform.python_version(), 'threads': {k:os.environ.get(k) for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS']}}
    (out / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=ex.SEED)
    records = []
    for fold, (fit_pos, eval_pos) in enumerate(cv.split(texts[train_idx], labels[train_idx]), 1):
        fit_idx, eval_idx = train_idx[fit_pos], train_idx[eval_pos]
        fit_text, eval_text = texts[fit_idx], texts[eval_idx]
        print(f'Fold {fold}/5: training BPE tokenizer on {len(fit_idx):,} reviews', flush=True)
        tok = build_tokenizer(fit_text)
        for name, min_df, ngram in SETTINGS:
            print(f'Fold {fold}/5: preparing {name}', flush=True)
            started = time.monotonic()
            vec = build_vectorizer(tok, min_df, ngram)
            X_fit, X_eval = vec.fit_transform(fit_text), vec.transform(eval_text)
            feature_time = time.monotonic() - started
            print(f'Fold {fold}/5: fitting {name}, {X_fit.shape[1]:,} features', flush=True)
            model = build_model()
            started = time.monotonic()
            with warnings.catch_warnings(record=True) as captured:
                warnings.simplefilter('always')
                model.fit(X_fit, labels[fit_idx])
            pred = model.predict(X_eval)
            result = {'fold': fold, 'configuration': name, 'macro_f1': f1_score(labels[eval_idx], pred, average='macro', zero_division=0),
                      'features': X_fit.shape[1], 'epochs': model.n_iter_,
                      'convergence_warning': any(issubclass(w.category, ConvergenceWarning) for w in captured),
                      'warnings': ' | '.join(str(w.message) for w in captured),
                      'feature_seconds': round(feature_time, 2), 'fit_and_predict_seconds': round(time.monotonic()-started, 2)}
            records.append(result)
            pd.DataFrame(records).to_csv(out / 'fold_scores.csv', index=False)
            print(json.dumps(result), flush=True)
            del X_fit, X_eval, vec, model
    scores = pd.DataFrame(records)
    table = scores.pivot(index='fold', columns='configuration', values='macro_f1')
    summary = scores.groupby('configuration').macro_f1.agg(['mean', 'std'])
    summary.to_csv(out / 'summary.csv')
    a, b = (table[name].to_numpy() for name, _, _ in SETTINGS)
    try:
        if np.var(b - a, ddof=1) == 0:
            raise ValueError('Identical fold differences give a degenerate posterior; more data or analysis required')
        with warnings.catch_warnings():
            warnings.simplefilter('error', RuntimeWarning)
            probs = np.asarray(baycomp.CorrelatedTTest(a, b, rope=ex.ROPE, runs=1).probs(), dtype=float)
        if not np.isfinite(probs).all():
            raise ValueError('Non-finite posterior; inspect fold scores')
        bayes = {'A': SETTINGS[0][0], 'B': SETTINGS[1][0], 'rope': ex.ROPE,
                 'P_A_better_by_more_than_rope': float(probs[0]), 'P_within_rope': float(probs[1]),
                 'P_B_better_by_more_than_rope': float(probs[2])}
    except (RuntimeWarning, ValueError, ZeroDivisionError) as err:
        bayes = {'note': str(err), 'A': SETTINGS[0][0], 'B': SETTINGS[1][0], 'rope': ex.ROPE}
    (out / 'bayesian_comparison.json').write_text(json.dumps(bayes, indent=2) + '\n')
    print('\nPaired fold scores:\n', table.to_string(), '\n\nSummary:\n', summary.to_string(), flush=True)
    print('\nBayesian comparison:', json.dumps(bayes), flush=True)
    print(f'Results saved to {out}. Final test has not been used.', flush=True)


if __name__ == '__main__':
    main()
