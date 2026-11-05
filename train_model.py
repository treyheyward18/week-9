"""Train a GroupEstimate model on the coffee review data and pickle it.

The data file (data/simplified_coffee.csv) is not committed to GitHub
(*.csv is in .gitignore). Only the pickled model is committed, and the
Streamlit app loads that.

Run from the repository folder:
    python train_model.py
"""
import pickle

import pandas as pd

from apputil import GroupEstimate

DATA_PATH = "data/simplified_coffee.csv"
MODEL_PATH = "coffee_price_model.pickle"
FEATURES = ["roast", "loc_country"]
TARGET = "100g_USD"


def main():
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=FEATURES + [TARGET])

    # median, because a few very expensive coffees skew the mean
    model = GroupEstimate(estimate="median")
    model.fit(df[FEATURES], df[TARGET], default_category="roast")

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    print(f"Saved model with {len(model.group_estimates_)} groups "
          f"to {MODEL_PATH}")


if __name__ == "__main__":
    main()
