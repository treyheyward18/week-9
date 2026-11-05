import pickle

import streamlit as st

from apputil import *


MODEL_PATH = "coffee_price_model.pickle"


@st.cache_resource
def load_model(path=MODEL_PATH):
    """Load the pickled GroupEstimate model once and reuse it."""
    with open(path, "rb") as f:
        return pickle.load(f)


model = load_model()

st.write(
    '''
# Week 9: Coffee Price Estimator

Pick a roast type and the country the roaster is located in. The app
returns the **median price per 100g (USD)** for coffees in that group,
learned from the coffee review data with the `GroupEstimate` model.
''')

# dropdown options come from the groups the model learned in fit()
groups = model.group_estimates_.index
roasts = sorted(groups.get_level_values("roast").unique())
countries = sorted(groups.get_level_values("loc_country").unique())

roast = st.selectbox(
    "Roast type", roasts, index=roasts.index("Medium-Light")
)
country = st.selectbox(
    "Roaster location", countries, index=countries.index("United States")
)

if st.button("Estimate price"):
    price = model.predict([[roast, country]])[0]

    if (roast, country) in groups:
        st.metric("Estimated price per 100g", f"${price:.2f}")
    elif price == price:  # not NaN, so the roast fallback was used
        st.metric("Estimated price per 100g", f"${price:.2f}")
        st.caption(
            f"No {roast} roasts from {country} in the data, so this is "
            f"the median for all {roast} roasts."
        )
    else:
        st.warning("Not enough data to estimate a price for this choice.")
