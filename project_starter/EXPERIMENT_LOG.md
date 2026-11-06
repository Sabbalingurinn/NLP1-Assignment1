# Project 1 experiment log

## Chronological study record

This timeline reconstructs the order from our work and the saved output. Exact start and finish times were not recorded, so they are not invented here. Details and tables follow below.

1. **Tutorial setup.** The old `hw9env` kernel pointed to a missing Python executable. Codex created the project-local `.venv` using Python 3.14 and installed `requirements.txt`; `import part1` was checked. The user was instructed to select `.venv` in VS Code.
2. **Tutorial learning.** We discussed segmentation, Heaps' law, BPE, probability tables, decision rules, feature representations, model evaluation, and log-linear models. Codex added code in the Heaps' law exercise to graph observed vocabulary growth against `sqrt(2) * n**0.5`. A log-log fit on cached AG News gave approximately `K=65.76, beta=0.450` when the `n=1` observation was excluded. That exercise was separate from the IMDb project.
3. **Project plan.** The user chose BPE, a bigram representation, five-fold validation, a 70/15/15 split, ROPE ±0.01 and an exploratory k-means approach. Codex added linear SVM as a third supervised architecture alongside logistic regression and a one-hidden-layer MLP, because k-means itself is unsupervised. The project checklist calls for the three feature representations, three architectures, cross-validation and Bayesian analysis.
4. **Dataset selection and preparation.** The user chose `mteb/imdb` and requested shuffled splits. The published train and test sets were combined to meet the requested 70/15/15 proportions, stratified by sentiment, and shuffled with seed 42. The new test partition has not been evaluated for this project. Pooling means this is not the official IMDb benchmark test split.
5. **Full baseline comparison.** The user ran the initial script, comparing 3 representations × 4 methods on five training folds and on the separate validation partition. The original script's 5-fold scores, validation scores, config and pairwise Bayesian outputs are preserved under `results/baseline/`. All five counts/logistic fold fits and the counts/logistic validation fit issued convergence warnings. The 64-unit bigram MLP led with mean macro F1 0.903438; see full table below. The initial discussion selected ±0.01 as ROPE, without yet establishing a dataset-specific reason why this size matters.
6. **Initial interpretation.** We explained macro F1, the feature encoders, the four model families, optimizer convergence and the Bayesian correlated-t comparison. Results were treated as development results, not final test performance. The user asked to stop using k-means in subsequent experiments; it was removed from the active script while its historical results were retained.
7. **Depth and trigram screening.** Codex ran one-fold comparisons of 64 vs 32–32 hidden layers using bigrams and hashed trigrams. Both architectures used the same feature matrix for a given representation. All four results are below and in `results/one_fold/`; hashing confounds the bigram/trigram comparison. During the 64-unit bigram rerun, Codex sent an interrupt just before the fit returned. The returned result may reflect interrupted training; the script captured warnings but only recorded convergence warnings, so we cannot verify that fit completed normally. Do not use it as a reliable converged comparison. The original fold-1 baseline is retained separately.
8. **BPE and feature screening.** At the user's request Codex tested a 2,000-token BPE vocabulary, `min_df=2`, unigram+bigram counts, and all changes together, against a matched capped baseline, all on the first fold. To manage resources, all configurations were limited to 100,000 features and used float32; both differences from the earlier baseline matter. Results are below and in `results/bigram_tuning/`. This was not a five-fold or final-test evaluation.
9. **One-fold screening conclusion.** The combined configuration scored 0.908372 against 0.906218 for its capped baseline on one fold; this was insufficient to establish an improvement.
10. **Paired five-fold follow-up.** At the user’s request, Codex evaluated the two highest-scoring 2,000-BPE candidates on the same five folds. The combined configuration averaged 0.905027 macro F1 versus 0.902287 for bigrams alone. The Bayesian comparison assigned 99.0% probability to a difference within the chosen ±0.01 ROPE, conditional on its assumptions. The separate validation and test partitions were not used. Details are at the end of this log.


## Aim and data

Classify IMDb movie reviews as positive or negative using `mteb/imdb`. The source had 24,904 published training and 24,678 published test reviews, for 49,582 reviews in total. There were 24,884 positive and 24,698 negative reviews. The user requested a shuffled 70/15/15 split; therefore the published splits were pooled and re-split with stratification and seed 42: 34,707 training, 7,437 validation and 7,438 held-out test reviews. Results do not use the official IMDb benchmark split and are not directly comparable with it. The final test has not been used for model assessment. Exact text duplicates within or between the published splits were not found in the initial inspection. Some reviews contain HTML-like `<br />` markers; no special removal was applied.

## First full comparison

The original results are in `results/baseline/` and are retained. Five shuffled, stratified folds within the training partition; each fold's BPE tokenizer and representation were trained on that fold's training data. Separate validation was evaluated after fitting on the full training partition. Metric: macro F1. BPE: byte-level, maximum vocabulary 1,000, minimum pair frequency 2, no lowercasing. Representations: binary BPE-token indicators, BPE-token counts, adjacent BPE-token bigram counts. Models: scikit-learn logistic regression (max_iter 500), MLP (one 64-unit hidden layer, ReLU, Adam, max_iter 100), linear SVM, and k-means with cluster labels determined from training labels only. Seed 42.

| Representation | Model | Mean five-fold macro F1 | Validation macro F1 |
|---|---|---:|---:|
| Bigrams | MLP (64) | 0.903438 | 0.901155 |
| Bigrams | Logistic regression | 0.892809 | 0.889323 |
| Bigrams | Linear SVM | 0.883649 | 0.879648 |
| Counts | Logistic regression | 0.832989 | 0.822764 |
| Counts | Linear SVM | 0.832207 | 0.822889 |
| Indicators | Linear SVM | 0.822154 | 0.812224 |
| Indicators | Logistic regression | 0.822010 | 0.811695 |
| Counts | MLP (64) | 0.813973 | 0.806870 |
| Indicators | MLP (64) | 0.799404 | 0.793852 |
| Indicators | K-means | 0.488614 | 0.485962 |
| Bigrams | K-means | 0.457815 | 0.456597 |
| Counts | K-means | 0.446781 | 0.440810 |

Counts/logistic reached the 500-iteration limit in every fold and on validation; its scores may reflect incomplete optimization. K-means was a weak exploratory clustering baseline and has been removed from future runs at the user's request; the historical results remain above.

The user chose a ROPE of ±0.01 macro F1. The `baycomp` correlated t comparison of paired fold scores for bigrams/logistic (A) and bigrams/MLP (B) gave P(A better by >0.01)=0.000033, P(within ROPE)=0.312702 and P(B better by >0.01)=0.687265. These values are conditional on the model and one five-fold run. Do not report the MLP as definitively superior; compare performance, uncertainty, and computational cost.

## Follow-up design: first fold only

Compare one hidden layer of 64 ReLU units with two hidden layers of 32 ReLU units each. Use the same first fold, training-only BPE and macro F1. Compare original BPE bigram counts with BPE trigram counts hashed into 65,536 nonnegative features to manage memory. Hash collisions are possible, so a bigram/trigram difference also reflects hashing and feature-space choices; this is not an isolated test of n-gram length. Both hidden-layer architectures are tested with each representation. Results are exploratory, not five-fold estimates; the held-out validation and test sets are not used. Repeat promising designs across all five folds before making a main claim. See `results/one_fold/` for settings and scores, including convergence and timing.

## AI assistance

Codex wrote the code for the experiment and helped organize the project's files and folders. The student remains responsible for understanding, checking and reporting the methods and results. Preserve this record and add further assistance when used.

### Follow-up results

The one-fold comparison completed on the same train/validation fold for all four systems. Separate validation and final test sets were not evaluated. Feature extraction was shared between the two architectures for each representation. Training plus prediction times below are indicative only and depend on the machine and other load.

| Representation | MLP hidden layers | Fold 1 macro F1 | Training iterations | Fit + prediction time |
|---|---|---:|---:|---:|
| Bigrams (223,384 features) | 64 | 0.907949 | 7 | 144.53 s |
| Bigrams (223,384 features) | 32, 32 | 0.904636 | 19 | 185.31 s |
| Hashed trigrams (65,536 buckets) | 64 | 0.876683 | 22 | 90.97 s |
| Hashed trigrams (65,536 buckets) | 32, 32 | 0.870485 | 18 | 37.60 s |

No convergence warning was recorded. However, Codex sent an interrupt during the 64-unit bigram fit just before it returned. Scikit-learn can catch an interruption and return a partially trained model, and the script did not preserve other warning messages. Treat the 0.907949 bigram baseline and comparisons using it as potentially interrupted, not as evidence of normal convergence. Feature preparation took 28.52 seconds for bigrams and 24.25 seconds for hashed trigrams, separately from the times above. The 32–32 bigram result was 0.904636; compare it cautiously with the original independent five-fold fold-1 score 0.90564, since the rerun baseline may have been interrupted. The hashed trigram two-layer score was lower than the hashed trigram one-layer score in this exploratory fold, but hashing/collisions and different feature-space sizes confound comparisons with bigrams. No five-fold or Bayesian inference is possible from this single fold.

The previous original bigram MLP fold-1 result was 0.90564. The 0.90795 rerun is not a clean reproducibility check because it may have been interrupted, as described above; any additional causes of differences are unknown. Both depth variants in the follow-up were fitted to the same prepared feature matrices within their representation.

## One-fold BPE and feature tuning

Following the user's request, screened a larger 2,000-token BPE vocabulary, rare-feature filtering, unigram+bigram counts, and the combination of all three changes. The same first fold and 64-unit ReLU MLP with Adam were used. To limit memory, **all** five configurations (including their baseline) were capped at the 100,000 most frequent features and used float32 matrices; this differs from the prior uncapped baseline. The same trained tokenizer was reused across settings with the same BPE vocabulary size. `min_df=2` requires a feature to appear in at least two training reviews, not simply twice overall; in unigram+bigram configurations it applies to both. Unigrams and bigrams compete for the shared feature cap. Training-only BPE/feature fitting; no separate validation or test usage. One fold only; no Bayesian or five-fold conclusion.

| Configuration | BPE max vocabulary | Minimum document frequency | N-grams | Features | Fold macro F1 | Fit+predict seconds |
|---|---:|---:|---|---:|---:|---:|
| Capped baseline | 1,000 | 1 | bigram | 100,000 | 0.906218 | 127.00 |
| 2,000 BPE | 2,000 | 1 | bigram | 100,000 | 0.906795 | 133.88 |
| Rare-feature filter | 1,000 | 2 | bigram | 100,000 | 0.905205 | 138.27 |
| Unigrams + bigrams | 1,000 | 1 | both | 100,000 | 0.902619 | 129.95 |
| All changes | 2,000 | 2 | both | 100,000 | 0.908372 | 129.97 |

No convergence warnings. Differences are small and exploratory: the combined setting exceeds the capped baseline by 0.002153 macro F1 (0.2153 percentage points) on this fold. All hit the feature cap, so the frequency filter did not reduce dimensionality under this setup. Filtered and unfiltered configurations need not have identical selected features. Test any proposed choice on all five folds and evaluate runtime/memory before claiming improvement. The earlier uncapped baseline and separate validation remain recorded above.

Code: `bigram_tuning.py`; machine-readable settings and results: `results/bigram_tuning/`. Codex designed, implemented, and ran this screening and updated the log.

## Training duration and stopping rules

The MLP uses scikit-learn Adam with `max_iter=100`, `tol=0.0001`, `n_iter_no_change=10`, and `early_stopping=False`. With early_stopping disabled, the stopping check is based on training loss, not the held-out fold. `max_iter` is an upper limit in epochs, not a promise to execute exactly 100. In the uninterrupted feature-tuning run, the selected combined configuration ran for 20 epochs and the capped baseline for 21, with no convergence warnings. More epochs may improve or worsen validation performance; a controlled training-length experiment needs matched fold comparisons and attention to overfitting. Running five cross-validation folds is a different question from training a single model for more epochs.

## Paired five-fold follow-up of the top two configurations

The user requested full five-fold cross-validation for the two highest-scoring candidates from the preceding one-fold tuning: (A) 2,000-token BPE, bigram counts, min_df=1; and (B) 2,000-token BPE, unigram+bigram counts, min_df=2. Both use a 100,000-feature cap, float32 matrices and the same 64-unit ReLU MLP/Adam as the screening experiment. Each fold has a single shared BPE tokenizer fitted only on its training reviews; the two feature vocabularies are independently fitted to those training reviews. Folds use the same shuffle/seed and training partition as earlier. The separate validation and test sets are not used. Record each model's macro F1, features, iterations, warnings and timing as it completes in `results/top2_cv/fold_scores.csv`. A paired correlated-t comparison with ROPE ±0.01 is calculated once all folds complete. This is a follow-up after screening on fold 1, so selection on that fold should be considered when interpreting results. The unchanged 1,000-BPE capped baseline is not one of the selected top two and is not re-evaluated in this run.

Running script: `top2_cross_validation.py`. Previous runs remain in their own result folders. The run completed; the results and warnings are recorded below.

### Paired five-fold results

| Fold | A: 2,000 BPE + bigrams | B: 2,000 BPE + min_df=2 + unigrams and bigrams | B − A |
|---|---:|---:|---:|
| 1 | 0.906795 | 0.908372 | +0.001577 |
| 2 | 0.903624 | 0.903486 | −0.000139 |
| 3 | 0.901167 | 0.908370 | +0.007204 |
| 4 | 0.899549 | 0.902741 | +0.003192 |
| 5 | 0.900298 | 0.902167 | +0.001869 |
| Mean | **0.902287** | **0.905027** | **+0.002740** |

Standard deviations across folds were 0.002950 for A and 0.003088 for B. All ten fits completed with no reported training warnings; the separate validation and test partitions were not used. Both configurations used exactly 100,000 features per fold. Feature building and model timing are retained per fit in `results/top2_cv/fold_scores.csv`.

For `baycomp` correlated-t with ROPE ±0.01 macro F1, the conditional posterior probabilities were P(A better by more than ROPE)=0.001173, P(within ROPE)=0.990183, and P(B better by more than ROPE)=0.008645. This suggests the difference is likely inside the chosen region under the analysis assumptions. The ROPE size needs a task-specific justification; the model assumes a particular correlated-fold distribution; and selecting both candidates using fold 1 makes that fold partially selected-on. These five-fold data do not prove equality or guarantee performance on new data. The earlier uncapped 1,000-BPE bigram MLP (mean 0.903438, separate validation 0.901155) is a differently constructed historical reference, not a directly paired, freshly rerun comparison with either capped candidate.

Codex created and ran the paired cross-validation script and wrote this update. Further experiment: if selecting B, check the separate validation set without revisiting it repeatedly, then consider final evaluation on the untouched test only once the choices are fixed.

## Project moved to Desktop

On the user's request, Codex moved `project_starter` (including code, results, cached IMDb data and this log) from the Downloads tutorial workspace to `Desktop/NLP1-IMDb-Project/project_starter`. The tutorial notebook, supporting code, and saved tutorial models were copied to `Desktop/NLP1-IMDb-Project/part1`; the original tutorial files remain in Downloads. A fresh Python 3.14 virtual environment was created in the Desktop folder with the same installed package versions, captured in `requirements-lock.txt`. Further work should use the Desktop folder as the VS Code workspace.

## Workspace organization and evaluation handoff

On 6 November 2026, Codex grouped the existing IMDb experiment outputs under `results/`, updated the project guide, and added a reusable evaluation command for the two five-fold candidates. The existing results were not changed or rerun. The new command shares the tokenizer, vectorizer and model construction with `top2_cross_validation.py`; it has not evaluated the IMDb validation or test split. The project's cached IMDb data was checked and its ordered text/label SHA-256 was recorded as `0ed0de756c57a4ac8868e7cc2a9970650e930d94cdb0347716d385c9e1e705be`; future IMDb loads check it and request dataset revision `d05f0155defa7991dad75bc68c5ccb6774b1fdc5`. This fingerprint was recorded during the cleanup, not during the original experiments. Codex also added guards against overwriting result directories and verified the workflow with synthetic examples.
