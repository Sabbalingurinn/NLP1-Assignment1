# IMDb experiment guide

The saved comparisons are complete; consult [the summary](PROJECT_SUMMARY.md) and [experiment log](EXPERIMENT_LOG.md) before deciding whether any additional evaluation is needed. Run the commands below from the project root. Scripts refuse to write into a nonempty output folder, so choose a fresh `--output` for each new run.

## Existing results

| Folder under `project_starter/results/` | What it contains |
| --- | --- |
| `baseline/` | Original five-fold comparison, validation scores and Bayesian comparisons. |
| `one_fold/` | Exploratory MLP-depth and bigram/hashed-trigram comparison. |
| `bigram_tuning/` | Exploratory first-fold BPE and feature screening. |
| `top2_cv/` | Paired five-fold comparison of the two leading screened configurations. |

These files are preserved study evidence, including historical k-means outputs. K-means is not included in new runs. Some original counts/logistic fits issued convergence warnings, and one exploratory bigram fit may have been interrupted; see the log.

## Data and reproducibility

The project pools the published train and test partitions of `mteb/imdb`, then creates a stratified 70/15/15 split using seed 42. This means results are not directly comparable to the dataset's official test benchmark. The saved data contains 49,582 reviews. Scripts request the cached dataset revision `d05f0155defa7991dad75bc68c5ccb6774b1fdc5` and verify the ordered text and label fingerprint, which was recorded during the workspace cleanup. If the data or its order has changed, they stop instead of silently constructing a different split.

The BPE tokenizer and feature vocabulary are fitted only on the training portion of each fold or evaluation. IMDb data is cached in `.hf_cache`; the environment is in the root `.venv` and its installed versions are in `requirements-lock.txt`. The cache can also be used offline, for example by setting `HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1` before a command.

## Optional follow-up evaluation

The five-fold follow-up compared a 2,000-token BPE tokenizer with either bigrams or combined unigrams/bigrams, a 100,000-feature cap, float32 and a 64-unit MLP. The combined version also uses `min_df=2`. **Neither follow-up candidate has been run on the separate validation or final test partition.** Choose a candidate based on the saved results and practical considerations; the Bayesian comparison found a high probability of being within the chosen ROPE under its assumptions.

If you choose to evaluate the combined candidate on the separate validation set, fit on training and score validation with:

```bash
.venv/bin/python project_starter/evaluate_candidate.py --candidate bpe2000_rare_unigrams_bigrams --stage validation
```

For the bigrams-only candidate, replace the candidate with `bpe2000_bigrams`. If you decide to use the final test, first settle the candidate and settings, then fit on training plus validation and evaluate the test once:

```bash
.venv/bin/python project_starter/evaluate_candidate.py --candidate bpe2000_rare_unigrams_bigrams --stage test
```

Both commands use the same feature and model construction as the paired five-fold script and save their configuration, macro F1, accuracy, per-class metrics, confusion matrix, warnings and timing in a new results folder. Running either on IMDb trains a model and can take significant time and memory. The test command should only be used for the candidate you have actually selected. The course checklist does not explicitly require a final test.

The older generic command `experiment.py --final ...` evaluates its own indicator, counts, bigram or hashed-trigram feature settings. It does **not** reproduce the capped combined unigram/bigram candidate; use `evaluate_candidate.py` for either five-fold follow-up candidate.

## Scripts

| Script | Purpose |
| --- | --- |
| `experiment.py` | General full comparison and optional final evaluation for its own feature settings. Accepts `--imdb`, `--csv PATH` or `--demo`. |
| `one_fold.py` | Exploratory first-fold MLP-depth and n-gram comparison. |
| `bigram_tuning.py` | Exploratory first-fold BPE and feature screening. |
| `top2_cross_validation.py` | Five-fold comparison of the two screened candidates. |
| `evaluate_candidate.py` | Optional validation or test assessment of either five-fold candidate. |

Each script supports `--help`. The `--demo` option uses synthetic data for a quick code check; demo scores are not IMDb results. A real full comparison can be expensive. Use a new `--output` directory to retain each run, and avoid repeatedly making choices based on the held-out test.

For a CSV with `experiment.py`, the default columns are `text` and `label`; use `--text-column` and `--label-column` for different names. Its `--final REPRESENTATION MODEL` choices are `indicator`, `counts`, `bigrams` or `trigrams_hashed`, paired with `logistic`, `mlp`, `mlp_32_32` or `svm`. For example, `--final bigrams mlp` uses the generic script's uncapped bigram representation, not the capped follow-up configuration.
