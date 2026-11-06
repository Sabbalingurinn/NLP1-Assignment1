"""First-fold, memory-limited screening of BPE and n-gram choices."""
import argparse
import json
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neural_network import MLPClassifier
from tokenizers import Tokenizer, models, pre_tokenizers, trainers

import experiment as ex


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--demo', action='store_true')
    parser.add_argument('--output', default='project_starter/results/new_bigram_tuning')
    args = parser.parse_args()
    out = ex.prepare_output_dir(args.output)
    source = argparse.Namespace(imdb=not args.demo, demo=args.demo, csv=None, text_column='text', label_column='label')
    df = ex.load_data(source)
    texts, labels = df.text.to_numpy(), df.label.to_numpy()
    train_idx, remaining = train_test_split(np.arange(len(df)), test_size=.30, stratify=df.label, random_state=ex.SEED)
    val_idx, test_idx = train_test_split(remaining, test_size=.5, stratify=df.label.iloc[remaining], random_state=ex.SEED)
    rng = np.random.default_rng(ex.SEED)
    train_idx, val_idx, test_idx = (rng.permutation(idx) for idx in (train_idx, val_idx, test_idx))
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=ex.SEED)
    fit_pos, eval_pos = next(cv.split(texts[train_idx], labels[train_idx]))
    fit_idx, eval_idx = train_idx[fit_pos], train_idx[eval_pos]
    X_text, V_text = texts[fit_idx], texts[eval_idx]
    config = {'source': 'synthetic demo' if args.demo else 'mteb/imdb pooled', 'seed': ex.SEED,
              'fold': 1, 'folds': 5, 'fit_size': len(fit_idx), 'eval_size': len(eval_idx),
              'metric': 'macro_f1', 'max_features': 100000, 'dtype': 'float32',
              'mlp_hidden_layers': [64], 'mlp_activation': 'relu', 'mlp_solver': 'adam', 'mlp_max_iter': 100,
              'min_df_meaning': 'minimum number of training documents containing a feature',
              'selection': 'one fold screening; separate validation and final test untouched'}
    (out / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
    configs = [
        ('capped_baseline', 1000, 1, (2, 2)),
        ('bpe_2000', 2000, 1, (2, 2)),
        ('rare_filter', 1000, 2, (2, 2)),
        ('unigrams_and_bigrams', 1000, 1, (1, 2)),
        ('all_changes', 2000, 2, (1, 2)),
    ]
    records = []
    # Train and reuse exactly the same BPE tokenizer for settings with the same vocabulary size.
    for vocab_size in (1000, 2000):
        print(f'Training BPE tokenizer: max vocabulary {vocab_size}', flush=True)
        tok = Tokenizer(models.BPE(unk_token='[UNK]'))
        tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
        trainer = trainers.BpeTrainer(vocab_size=vocab_size, min_frequency=2, special_tokens=['[UNK]'],
                                      initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False)
        tok.train_from_iterator(X_text, trainer=trainer)
        for name, vs, min_df, ngram in configs:
            if vs != vocab_size:
                continue
            print(f'Building {name}: BPE={vs}, min_df={min_df}, ngrams={ngram}', flush=True)
            started = time.monotonic()
            vectorizer = CountVectorizer(tokenizer=lambda text: tok.encode(text).tokens, token_pattern=None,
                                         lowercase=False, ngram_range=ngram, min_df=min_df,
                                         max_features=100000, dtype=np.float32)
            X = vectorizer.fit_transform(X_text)
            V = vectorizer.transform(V_text)
            feature_time = time.monotonic() - started
            print(f'Fitting {name}: {X.shape[1]:,} features', flush=True)
            clf = MLPClassifier(hidden_layer_sizes=(64,), max_iter=100, random_state=ex.SEED)
            started = time.monotonic()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always', ConvergenceWarning)
                clf.fit(X, labels[fit_idx])
            pred = clf.predict(V)
            result = {'configuration': name, 'vocab_size': vs, 'min_df': min_df,
                      'ngram_range': str(ngram), 'features': X.shape[1],
                      'macro_f1': f1_score(labels[eval_idx], pred, average='macro', zero_division=0),
                      'epochs': clf.n_iter_,
                      'convergence_warning': any(issubclass(w.category, ConvergenceWarning) for w in caught),
                      'feature_seconds': round(feature_time, 2), 'fit_and_predict_seconds': round(time.monotonic()-started, 2)}
            records.append(result)
            pd.DataFrame(records).to_csv(out / 'scores.csv', index=False)
            print(json.dumps(result), flush=True)
            del X, V, vectorizer, clf
    print('Finished. Results are one-fold screening, not final test results.')


if __name__ == '__main__':
    main()
