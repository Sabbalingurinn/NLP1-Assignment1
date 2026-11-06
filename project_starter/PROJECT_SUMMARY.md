# IMDb project: results and preparation notes

## What was studied

The task is binary sentiment classification of IMDb reviews. The cached [`mteb/imdb` dataset](https://huggingface.co/datasets/mteb/imdb) contains 49,582 reviews: 24,884 positive and 24,698 negative. Its dataset card asks users to cite both the [original IMDb dataset paper](https://aclanthology.org/P11-1015/) and [MTEB](https://github.com/embeddings-benchmark/mteb). The published train and test partitions were pooled, shuffled and stratified into 34,707 training, 7,437 validation and 7,438 held-out test reviews using seed 42. Pooling was a deliberate choice to obtain a 70/15/15 split; the resulting test is not the official benchmark test, so benchmark scores are not directly comparable.

The primary metric is macro F1: compute F1 for each sentiment class and average them equally. Five-fold cross-validation is performed within the training portion. In each fold, the byte-level BPE tokenizer and feature vocabulary are fitted only on that fold's training data, and models are scored on its held-out fold. The separate validation set was used for the original baseline grid; the separate test set has not been evaluated.

## Course checklist

| Requirement | Work already recorded |
| --- | --- |
| Three representations | BPE-token indicators, token counts and adjacent BPE-token bigram counts. |
| Three model architectures | Logistic regression, a single-hidden-layer MLP and a linear SVM. A deeper MLP was also explored on one fold. |
| Cross-validation and metric | Five stratified training folds; macro F1. |
| Bayesian estimation and ROPE | Paired correlated-t comparisons of fold scores, using a ROPE of ±0.01 macro F1. |

Historical k-means results are also saved as an exploratory unsupervised baseline; it is not a substitute for the supervised architectures above. The current full-grid script has since been extended with more feature/model choices and no longer runs k-means, so it is not an exact rerun of the original grid.

## Main findings

In the original comparison, the best system by mean cross-validation macro F1 was BPE bigrams with a 64-unit MLP: **0.903438** across the five folds and **0.901155** on the separate validation set. Bigrams with logistic regression scored 0.892809 across folds; bigrams with a linear SVM scored 0.883649. Counts/logistic fits issued convergence warnings, so those scores need caution.

After one-fold screening, the follow-up compared two capped, 2,000-token-BPE MLP systems over the same five folds:

| Features | Mean macro F1 | Fold standard deviation |
| --- | ---: | ---: |
| Bigrams, `min_df=1` | 0.902287 | 0.002950 |
| Unigrams + bigrams, `min_df=2` | 0.905027 | 0.003088 |

Both use at most 100,000 features, float32 matrices and a single 64-unit hidden layer. Their difference was 0.002740 macro F1. The paired Bayesian analysis assigned approximately **99.0%** probability to the difference being inside the chosen ±0.01 ROPE, conditional on the model's assumptions. This does not prove the systems equal. The two candidates were selected using fold 1, and the separate validation and test sets were not used in this follow-up. The original uncapped baseline has different settings and is not a freshly paired comparison with these two systems.

The one-fold depth and trigram study is exploratory. Its 64-unit bigram rerun may have been interrupted; the hashed-trigram comparison also changes the feature construction and can have hash collisions. Do not treat it as a clean, five-fold comparison. The full record and per-fold results are in [the experiment log](EXPERIMENT_LOG.md) and `results/`.

## Finish the write-up and oral preparation

- Choose a system and explain the trade-offs using the results above; the highest observed mean alone is not proof of a meaningful improvement.
- Explain why ±0.01 macro F1 would count as a practically negligible difference for this task, or acknowledge that its size was chosen provisionally and remains to be justified. The comparison's interpretation depends on that choice.
- Be ready to explain the data split and pooling, BPE and features, each architecture, macro F1, cross-validation, and the Bayesian result's limits. The assignment allows up to five backup slides on data/metrics, features, architectures, results and extra material.
- Retain the dataset source, settings, results and AI-assistance record for the later blog. Codex wrote the experiment code and helped organize the project's files and folders.
- A separate test evaluation is not explicitly listed as a requirement in the assignment. If you choose to add validation or final-test evaluation, first read [the experiment guide](README.md); it describes the correct commands and the remaining unused partitions.
