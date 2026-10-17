"""Memes folds melanges et meme seed pour toutes les experiences."""

from sklearn.model_selection import KFold, cross_validate


def evaluate(model, X, y, folds=5, seed=42):
    splitter = KFold(n_splits=folds, shuffle=True, random_state=seed)
    scores = cross_validate(
        model, X, y, cv=splitter, n_jobs=1,
        scoring={
            "rmse": "neg_root_mean_squared_error",
            "mae": "neg_mean_absolute_error",
            "medae": "neg_median_absolute_error",
        },
        error_score="raise",
    )
    return {metric: (-scores[f"test_{metric}"]).tolist() for metric in ("rmse", "mae", "medae")}
