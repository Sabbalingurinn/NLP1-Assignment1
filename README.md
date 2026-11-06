# NLP1 IMDb sentiment project

This workspace contains the completed comparison experiments, their saved results, and the original course tutorial. Start with [the project summary](project_starter/PROJECT_SUMMARY.md) for the findings and [the experiment log](project_starter/EXPERIMENT_LOG.md) for the full record.

The course checklist's required comparisons have been run: indicator, count and bigram features; logistic regression, a one-layer MLP and a linear SVM; five-fold cross-validation; and Bayesian comparisons with a ROPE. There is also a follow-up comparison of two tuned MLP configurations. The held-out test has **not** been evaluated. It is not explicitly required in the project checklist; the follow-up candidates also have not been evaluated on the separate validation set.

To finish preparing the project, choose and explain a system using the existing results, motivate the ±0.01 ROPE, and prepare for the oral assessment. The assignment permits up to five backup slides; see [the project instructions](part1/project.ipynb). Keep the experiment log and AI-assistance record for the eventual blog post. See [the project guide](project_starter/README.md) if you decide that an additional validation or final test evaluation is useful.

## Find your way around

| Location | Contents |
| --- | --- |
| `project_starter/PROJECT_SUMMARY.md` | Methods, main results, caveats and remaining decisions. |
| `project_starter/EXPERIMENT_LOG.md` | Chronological study and AI-assistance record. |
| `project_starter/results/` | Saved baseline, one-fold screening and five-fold follow-up outputs. |
| `project_starter/*.py` | Experiment scripts and an optional evaluator for the two follow-up candidates. |
| `project_starter/.hf_cache/` | Local IMDb cache, created when needed and excluded from Git. Keep it for offline work. |
| `part1/` | Course project instructions, tutorial notebook, support code and tutorial models. |
| `requirements.txt`, `requirements-lock.txt` | Course dependencies and the versions installed in this environment. |
| `.venv/` | Local Python 3.14 environment, excluded from Git. |

## Open the project

Open **NLP1-IMDb-Project** as a folder in VS Code. For notebooks, select the interpreter at `.venv/bin/python`. Open `part1/project.ipynb` for the assignment or `part1/tutorial.ipynb` for the tutorial; the tutorial expects its working directory to be `part1` because its example files and saved models are there.

In a terminal at the project root, check the available commands without training a model:

```bash
.venv/bin/python project_starter/evaluate_candidate.py --help
```

The original workspace already has the environment and cached dataset. A fresh Git clone does not include either: with Python 3.14, run `python3.14 -m venv .venv` and `.venv/bin/python -m pip install -r requirements-lock.txt`. An IMDb command will download the pinned dataset revision on first use. Installation and downloading an absent dataset require internet access; locked packages may also depend on the operating system and Python version.
