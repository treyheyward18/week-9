import numpy as np
import pandas as pd


class GroupEstimate:
    "Predict a continuous value from the group a row of categories falls in."

    VALID_ESTIMATES = ("mean", "median")

    def __init__(self, estimate="mean"):
        if estimate not in self.VALID_ESTIMATES:
            raise ValueError(
                f"estimate must be one of {self.VALID_ESTIMATES}, "
                f"got {estimate!r}"
            )
        self.estimate = estimate

        # learned during fit()
        self.columns_ = None
        self.group_estimates_ = None
        self.default_category = None
        self.default_estimates_ = None

    def fit(self, X, y, default_category=None):
        """Learn the estimate of ``y`` for each combination of categories.

        Parameters
        ----------
        X : pandas.DataFrame
            Categorical feature columns.
        y : array-like
            Continuous target values, same length as ``X``, no missing values.
        default_category : str, optional
            A single column of ``X`` to fall back on when a combination of
            categories was never seen during fitting.
        """
        X = pd.DataFrame(X)
        y = np.asarray(y)

        if y.ndim != 1:
            raise ValueError("y must be 1-dimensional.")
        if len(X) != len(y):
            raise ValueError("X and y must have the same number of rows.")
        if pd.isna(y).any():
            raise ValueError("y must not contain missing values.")

        # combine X and y into one (temporary) dataframe
        df = X.copy()
        df["_target"] = y  # positional, so index alignment can't misfire

        # one estimate per observed combination of categories
        self.columns_ = list(X.columns)
        self.group_estimates_ = (
            df.groupby(self.columns_, observed=True)["_target"]
            .agg(self.estimate)
        )

        # bonus: fallback estimates for a single category
        if default_category is not None:
            if default_category not in self.columns_:
                raise ValueError(
                    f"default_category {default_category!r} is not a "
                    f"column of X: {self.columns_}"
                )
            self.default_category = default_category
            self.default_estimates_ = (
                df.groupby(default_category, observed=True)["_target"]
                .agg(self.estimate)
            )
        else:
            self.default_category = None
            self.default_estimates_ = None

        return self

    def predict(self, X_):
        """Return the learned estimate for each row of ``X_`` as a NumPy array.

        Rows whose combination of categories was not seen in fit() get NaN
        (or the ``default_category`` estimate, if one was given to fit()).
        """
        if self.group_estimates_ is None:
            raise RuntimeError("Call fit() before predict().")

        if isinstance(X_, pd.DataFrame):
            X_ = X_[self.columns_]
        else:
            X_ = pd.DataFrame(X_, columns=self.columns_)

        # look up each row's group in the learned estimates
        if len(self.columns_) == 1:
            keys = pd.Index(X_[self.columns_[0]])
        else:
            keys = pd.MultiIndex.from_frame(X_)
        preds = self.group_estimates_.reindex(keys).to_numpy(
            dtype=float, copy=True
        )

        missing = np.isnan(preds)
        n_missing = int(missing.sum())

        if n_missing and self.default_estimates_ is not None:
            fallback = (
                self.default_estimates_
                .reindex(X_[self.default_category])
                .to_numpy(dtype=float)
            )
            preds[missing] = fallback[missing]
            print(
                f"{n_missing} observation(s) had an unseen category "
                f"combination; used the {self.default_category!r} "
                f"estimate instead."
            )
            missing = np.isnan(preds)
            n_missing = int(missing.sum())

        if n_missing:
            print(
                f"{n_missing} observation(s) belong to a group missing from "
                f"the training data; returning NaN for them."
            )

        return preds
