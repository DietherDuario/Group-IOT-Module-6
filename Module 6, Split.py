"""Module 6 - Split.

Splits Module 5's data by day (round 1 = train, round 2 = test)
and saves each set as a .npy file.
"""

from pathlib import Path

import numpy as np

TRAIN_ROUND = 1
TEST_ROUND = 2


def split_and_save(X, y, rounds, out_dir="data/split"):
    X, y, rounds = np.asarray(X), np.asarray(y), np.asarray(rounds)
    _validate(X, y, rounds)

    # True/False mask for each window: is it from round 1 or round 2?
    train = rounds == TRAIN_ROUND
    test = rounds == TEST_ROUND

    # The masks pick out only the matching windows
    sets = {
        "X_train": X[train],
        "y_train": y[train],
        "X_test": X[test],
        "y_test": y[test],
    }
    _check_labels(sets["y_train"], sets["y_test"])

    # Create the output folder if it doesn't exist yet
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    paths = {}
    for name, arr in sets.items():
        path = out / f"{name}.npy"
        np.save(path, arr, allow_pickle=False)  # no pickle = safer to load
        paths[name] = str(path)

    print(f"train: {train.sum()} windows | test: {test.sum()} windows")
    return paths


def _validate(X, y, rounds):
    # Every window needs exactly one label and one round
    if not (len(X) == len(y) == len(rounds)):
        raise ValueError(f"Length mismatch: X={len(X)}, y={len(y)}, rounds={len(rounds)}")
    if len(X) == 0:
        raise ValueError("Empty dataset from Module 5.")

    # Only rounds 1 and 2 are allowed, and both must be there
    unknown = set(np.unique(rounds)) - {TRAIN_ROUND, TEST_ROUND}
    if unknown:
        raise ValueError(f"Unexpected round values: {sorted(unknown)}")
    for r in (TRAIN_ROUND, TEST_ROUND):
        if not np.any(rounds == r):
            raise ValueError(f"No windows for round {r}.")


def _check_labels(y_train, y_test):
    train_labels, test_labels = set(np.unique(y_train)), set(np.unique(y_test))

    # Model can't be tested on an activity it never trained on
    if test_labels - train_labels:
        raise ValueError(f"Test has unseen labels: {sorted(test_labels - train_labels)}")

    # Not fatal, but that activity won't be tested
    if train_labels - test_labels:
        print(f"Warning: no test data for: {sorted(train_labels - test_labels)}")


if __name__ == "__main__":
    # Quick test with fake data
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 128, 9))
    y = np.tile(["sit", "walk"], 20)
    rounds = np.repeat([1, 2], 20)

    paths = split_and_save(X, y, rounds, out_dir="split_test")

    # Check the saved file loads back to the exact same data
    assert np.array_equal(np.load(paths["X_train"]), X[rounds == 1])
    print("save/load round-trip OK")