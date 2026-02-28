# Process Mining Project: Providing Recommendations

A predictive and prescriptive process monitoring system based on decision trees with boolean encoding. Given a production event log, the system predicts whether ongoing process instances will result in a positive (fast) or negative (slow) outcome and generates actionable recommendations for negatively predicted traces.


---

## Project Structure

```
my_implementation/
├── src/
│   ├── __init__.py
│   ├── types.py              # Data structures: BooleanCondition, ThresholdCondition, Path
│   ├── preprocessing.py      # Log import, prefix generation with padding, boolean encoding
│   ├── tree_utils.py         # DFS traversal, path compliance, missing condition extraction
│   └── recommendations.py    # extract_recommendations() and evaluate_recommendations()
├── project.ipynb   # Complete step-by-step notebook (main entry point)
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

The XES log files are expected in the parent directory:
```
../Production_avg_dur_training_0-80.xes   # Training set (80%, 177 traces)
../Production_avg_dur_testing_80-100.xes  # Test set (20%, 43 traces)
```

---

## Requirements

- Python 3.10+
- Dependencies listed in `requirements.txt`

### Installation

```bash
pip install -r requirements.txt
```

The main dependencies are:
- **pm4py** — process mining (XES import, EventLog handling)
- **scikit-learn** — decision tree classifier, GridSearchCV, metrics
- **pandas / numpy** — data manipulation
- **matplotlib / seaborn** — visualization
- **graphviz** — decision tree rendering (requires the `graphviz` system package)

---

## How to Run

Open and execute the notebook sequentially:

```bash
cd my_implementation
jupyter notebook tutorial_completo.ipynb
```

Alternatively, open the notebook in VS Code with the Jupyter extension.

---

## Pipeline Overview

The notebook (`tutorial_completo.ipynb`) executes the full pipeline:

| Step | Description |
|------|-------------|
| 0 | Import libraries |
| 1 | Load training and test XES logs |
| 2 | Generate prefix traces with padding |
| 3 | Extract unique activity names from the training log |
| 4 | Boolean encoding of prefix traces |
| 5 | Train a Decision Tree with GridSearchCV hyperparameter optimization |
| 6 | Evaluate the model (accuracy, precision, recall, F1, confusion matrix) |
| 7 | Extract recommendations for negatively predicted traces |
| 8 | Analyze the generated recommendations |
| 9 | Detailed example of a single recommendation |
| 10 | Evaluate recommendations against full traces (TP/TN/FP/FN, F-measure) |
| 11 | Visual example — highlight recommendation path on the decision tree |
| 12 | Compare prediction and recommendation metrics across prefix lengths 5 and 11 |
| 12b | Comprehensive prefix length analysis (1–12) with plots |
| 13 | Export all figures for the LaTeX report |

---

## Key Functions

### `extract_recommendations(tree, feature_names, class_values, prefix_set)`

Extracts recommendations from a trained decision tree for traces with a negative prediction. For each negatively predicted prefix trace, the function:

1. Extracts all positive paths from the tree via DFS
2. Filters paths compliant with the trace's current state
3. Selects the path with the highest confidence (shortest length as tiebreaker)
4. Returns the missing conditions as recommendations

**Returns:** a dictionary mapping each trace (as a frozenset of present activities) to a set of `BooleanCondition` objects (empty if no recommendation is needed or possible).

### `evaluate_recommendations(test_set, recommendations)`

Evaluates recommendations by checking whether the recommended activities were actually followed in the complete (non-truncated) test traces. The function follows the assignment-required signature where `test_set` is a dictionary containing:

- `'tree'`: the trained DecisionTreeClassifier
- `'feature_names'`: list of feature column names
- `'class_values'`: `['true', 'false']`
- `'prefix_test_set'`: boolean-encoded prefix test traces
- `'full_test_set'`: boolean-encoded full test traces

**Classification criteria:**
- **TP**: recommendation followed AND ground truth is positive
- **TN**: recommendation NOT followed AND ground truth is negative
- **FP**: recommendation followed AND ground truth is negative
- **FN**: recommendation NOT followed AND ground truth is positive

**Returns:** a dictionary with `tp`, `tn`, `fp`, `fn`, `precision`, `recall`, `f1_score`, `accuracy`, `evaluated`.

---

## Helper Modules

### `preprocessing.py`
- `import_log(file_path)` — loads an XES file into an EventLog
- `create_prefixes_log(log, prefix_length, use_padding)` — truncates traces and pads shorter ones
- `get_activity_names(log)` — extracts all unique activity names
- `boolean_encode(log, activity_names)` — converts an EventLog into a boolean-encoded DataFrame
- `exclude_keys_from_trace(trace_dict)` — filters out metadata keys, keeping only True activities

### `tree_utils.py`
- `get_positive_paths(tree, feature_names, positive_class_idx)` — DFS extraction of all positive-leaf paths
- `contradicts(condition, trace_dict)` — checks if a condition contradicts a trace's state
- `get_compliant_paths(paths, current_conditions)` — filters paths compatible with the trace
- `get_missing_conditions(path, current_conditions)` — extracts conditions not yet satisfied

### `types.py`
- `BooleanCondition` — represents an activity presence/absence condition
- `ThresholdCondition` — represents a numerical threshold condition
- `Condition` / `Path` — type aliases

---

## Results Summary

With the optimized decision tree (prefix length = 11):

| Metric | Decision Tree | Recommendations |
|--------|:------------:|:---------------:|
| Accuracy | 0.814 | 0.714 |
| Precision | 0.833 | 0.500 |
| Recall | 0.938 | 0.500 |
| F1-Score | 0.882 | 0.500 |

The prefix length comparison (1–12) shows that prediction quality is relatively stable across prefix lengths, while the recommendation system only becomes effective from prefix length 9 onward, with the best performance at prefix length 11.

---

## Report

The LaTeX report is in `../LaTeXTemplates_diaz-essay_v2/report_part1.tex` and provides a detailed description of the methodology, algorithms (with pseudocode), results and discussion.
