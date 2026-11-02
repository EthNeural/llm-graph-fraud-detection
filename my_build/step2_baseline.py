"""
Step 2: Fix the evaluation protocol, then set the bar every later model must beat.

Run:  python my_build/step2_baseline.py
Needs common.py finished first. Check answers at the bottom.

Two baselines:
  1. Majority class  -> the "do nothing" floor
  2. TF-IDF + Logistic Regression  -> text only, no graph, no LLM.
     If an LLM or GNN can't beat this, it isn't earning its cost.
"""
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from common import load, get_splits, isolated_mask, evaluate, best_threshold, RESULTS_DIR


def majority_baseline(y, train, test, iso_test):
    # TODO 6: find the most common label in y[train], predict it for every test node,
    # and return evaluate("majority", ...). No y_score needed.
    majority = np.bincount(y[train]).argmax()
    y_pred = np.full(len(test), majority)
    return evaluate("majority", y[test], y_pred, iso=iso_test)


def tfidf_baseline(x_text, y, train, val, test, iso_test, class_weight=None):
    # TODO 7:
    #   a) TfidfVectorizer(max_features=20000, ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    #   b) fit_transform on TRAIN texts only, transform the TEST texts
    #      THINK: why must you never fit the vectorizer on test text?
    #   c) LogisticRegression(max_iter=2000, class_weight=class_weight), fit on train
    #   d) y_pred = predict, y_score = predict_proba(...)[:, 1]
    #   e) return evaluate(f"tfidf ({class_weight})", ...)
    vec = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    X_train = vec.fit_transform(x_text[train])
    X_val = vec.transform(x_text[val])
    X_test = vec.transform(x_text[test])

    clf = LogisticRegression(max_iter=2000, class_weight=class_weight)
    clf.fit(X_train, y[train])
    t = best_threshold(y[val], clf.predict_proba(X_val)[:, 1])     # cut-off chosen on val
    y_score = clf.predict_proba(X_test)[:, 1]
    y_pred = (y_score > t).astype(int)
    return evaluate(f"tfidf ({class_weight})", y[test], y_pred, y_score, iso_test)


def main():
    x_text, edge_index, y = load()
    train, val, test = get_splits(y)
    iso_test = isolated_mask(edge_index, len(y))[test]
    print(f"train {len(train)}  val {len(val)}  test {len(test)}  "
          f"(isolated in test: {iso_test.sum()})")

    # TODO 8: run majority, tfidf with class_weight=None, and tfidf with "balanced".
    # Collect the returned dicts and json.dump them to RESULTS_DIR / "step2_baselines.json"
    # (indent=2). Every later step will add its own results file next to this one.
    results = [
        majority_baseline(y, train, test, iso_test),
        tfidf_baseline(x_text, y, train, val, test, iso_test, class_weight=None),
        tfidf_baseline(x_text, y, train, val, test, iso_test, class_weight="balanced"),
    ]
    with open(RESULTS_DIR / "step2_baselines.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved -> {RESULTS_DIR / 'step2_baselines.json'}")


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# CHECK YOUR ANSWERS (seed 42, to 3 decimals):
#   train 6803  val 2268  test 2268   (isolated in test: 385)
#   class-1 share in train / val / test: 0.363 each   <-- proves stratify worked
#
#                     acc    macroF1  F1_class1  AUROC  | conn mF1  iso mF1
#   majority         0.637   0.389    0.000       -     |  0.395    0.359
#   tfidf (None)     0.660   0.536    0.297     0.593   |  0.535    0.528
#   tfidf (balanced) 0.605   0.576    0.465     0.592   |  0.575    0.567
#
# THINK ABOUT (study guide):
#   - Majority gets 63.7% accuracy and catches ZERO class-1 users. What does that tell
#     you about reporting accuracy in a fraud paper?
#   - "balanced" LOWERS accuracy but RAISES macro-F1 and class-1 F1. Why? Which would a
#     fraud team prefer?
#   - AUROC is only ~0.59. Text alone is weak, and step 1 showed the graph is barely
#     better than random. That's the real research problem: two weak signals.
#     Can an LLM + graph combination get more out of them than either alone?
# ---------------------------------------------------------------------------
