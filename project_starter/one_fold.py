"""Exploratory first-fold comparison of MLP depth and BPE n-grams."""
import argparse
import json
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neural_network import MLPClassifier

import experiment as ex


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--demo', action='store_true', help='Run the same workflow on synthetic data, not IMDb')
    parser.add_argument('--output', default='project_starter/results/new_one_fold')
    args = parser.parse_args()
    out = ex.prepare_output_dir(args.output)
    source = argparse.Namespace(imdb=not args.demo, demo=args.demo, csv=None,
                                text_column='text', label_column='label')
    df = ex.load_data(source)
    texts, labels = df.text.to_numpy(), df.label.to_numpy()
    train_idx, other_idx = train_test_split(np.arange(len(df)), test_size=0.30,
                                           stratify=df.label, random_state=ex.SEED)
    val_idx, test_idx = train_test_split(other_idx, test_size=0.5,
                                         stratify=df.label.iloc[other_idx], random_state=ex.SEED)
    rng = np.random.default_rng(ex.SEED)
    train_idx, val_idx, test_idx = (rng.permutation(x) for x in (train_idx, val_idx, test_idx))
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=ex.SEED)
    fit_pos, eval_pos = next(splitter.split(texts[train_idx], labels[train_idx]))
    fit_idx, eval_idx = train_idx[fit_pos], train_idx[eval_pos]
    config = {'source': 'synthetic demo' if args.demo else 'mteb/imdb pooled official splits',
              'seed': ex.SEED, 'fold': 1, 'n_splits': 5, 'metric': 'macro_f1',
              'vocab_size': 1000, 'bpe_min_frequency': 2,
              'hashed_trigram_features': 2**16, 'hashed_trigram_alternate_sign': False,
              'hidden_layer_sizes': [[64], [32, 32]], 'max_iter': 100,
              'fold_train_size': len(fit_idx), 'fold_validation_size': len(eval_idx),
              'note': 'One-fold exploration; test and separate validation were not used.'}
    (out / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
    results = []
    for representation in ('bigrams', 'trigrams_hashed'):
        started = time.monotonic()
        print(f'Preparing {representation} features...', flush=True)
        X_fit, X_eval = ex.build_features(texts[fit_idx], texts[eval_idx], representation, 1000)
        feature_seconds = time.monotonic() - started
        print(f'Features: {X_fit.shape[1]:,}; preparation: {feature_seconds:.1f}s', flush=True)
        for hidden in ((64,), (32, 32)):
            print(f'Training {representation}, hidden layers {hidden}...', flush=True)
            model = MLPClassifier(hidden_layer_sizes=hidden, max_iter=100, random_state=ex.SEED)
            started = time.monotonic()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always', ConvergenceWarning)
                model.fit(X_fit, labels[fit_idx])
            pred = model.predict(X_eval)
            result = {'representation': representation, 'hidden_layers': str(hidden),
                      'macro_f1': f1_score(labels[eval_idx], pred, average='macro', zero_division=0),
                      'n_features': X_fit.shape[1], 'iterations': model.n_iter_,
                      'convergence_warning': any(issubclass(w.category, ConvergenceWarning) for w in caught),
                      'feature_seconds': round(feature_seconds, 2), 'fit_and_predict_seconds': round(time.monotonic()-started, 2)}
            results.append(result)
            pd.DataFrame(results).to_csv(out / 'scores.csv', index=False)
            print(json.dumps(result), flush=True)
        del X_fit, X_eval
    print('\nOne-fold exploration only; repeat promising choices across all five folds before drawing conclusions.')


if __name__ == '__main__':
    main()
