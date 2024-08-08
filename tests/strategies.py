import pandas as pd
from hypothesis import strategies as st

ascii_upper_letters = st.characters(categories=("Lu",), codec="ascii")


@st.composite
def events_series(draw, **kwargs):
    list_of_events = st.lists(st.text(alphabet=ascii_upper_letters,
                                      max_size=4,
                                      min_size=1), min_size=1)
    return pd.Series(draw(list_of_events))

@st.composite
def event_probs(draw, **kwargs):
    length = draw(st.integers(min_value=1, max_value=100))
    list_of_events = st.lists(st.text(alphabet=ascii_upper_letters,
                                      max_size=4,
                                      min_size=1),
                              min_size=length,
                              max_size=length,
                              unique=True)
    list_of_integers = st.lists(st.integers(min_value=1, max_value=3000),
                              min_size=length,
                              max_size=length)
    probs = pd.Series(draw(list_of_integers), index=draw(list_of_events))
    total = probs.sum()
    probs = probs / total
    return probs
