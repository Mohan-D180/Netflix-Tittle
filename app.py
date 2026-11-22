import datetime as dt
import joblib, pandas as pd, streamlit as st
from common import CLASSES, FEATURES, build_features

st.set_page_config(page_title="Netflix Maturity Predictor", page_icon="🎬", layout="wide")

@st.cache_resource
def load():
    return joblib.load('model.joblib')
bundle = load(); model = bundle['model']

COLORS = {'Kids': '#e08a3c', 'Teens': '#2a9d8f', 'Mature': '#b20710'}
EXAMPLES = {
    'Custom (start blank)': {},
    "Kids cartoon series": dict(type='TV Show', genres=["Kids' TV"], country='United States', seasons=3, year=2019,
        desc="A cheerful group of animal friends learn about sharing and kindness while exploring their colorful town."),
    "Crime thriller movie": dict(type='Movie', genres=['Thrillers', 'Dramas'], country='United States', minutes=112, year=2018,
        desc="A detective hunts a violent killer through a corrupt city while drug lords and dirty cops close in."),
    "Stand-up special": dict(type='Movie', genres=['Stand-Up Comedy'], country='United States', minutes=70, year=2020,
        desc="A comedian delivers raw, unfiltered jokes about sex, politics and family in a live show."),
}

st.title("🎬 Netflix Maturity Predictor")
st.caption("Enter a title's details and the model (tuned Gradient Boosting from the lab) predicts whether it is rated for **Kids**, **Teens** or **Mature** audiences.")

ex = EXAMPLES[st.sidebar.selectbox("Load an example", list(EXAMPLES))]
st.sidebar.markdown("---")
st.sidebar.markdown(f"**Model hold-out performance**  \nAccuracy: {bundle['metrics']['accuracy']:.3f}  \nMacro-F1: {bundle['metrics']['macro_f1']:.3f}")
st.sidebar.caption("Trained on the 2021 Netflix catalogue snapshot (Kaggle). Predictions are estimates, not official ratings.")

left, right = st.columns([3, 2], gap="large")
with left:
    st.subheader("1. Title details")
    c1, c2 = st.columns(2)
    types = ['Movie', 'TV Show']
    ctype = c1.radio("Type", types, index=types.index(ex.get('type', 'Movie')), horizontal=True)
    title = c2.text_input("Title (optional)", value="My new title")
    c3, c4 = st.columns(2)
    if ctype == 'Movie':
        minutes = c3.number_input("Duration (minutes)", 1, 400, ex.get('minutes', 100)); duration = f"{minutes} min"
    else:
        seasons = c3.number_input("Number of seasons", 1, 30, ex.get('seasons', 1)); duration = f"{seasons} Season" + ('s' if seasons > 1 else '')
    countries = bundle['countries']
    country = c4.selectbox("Main production country", countries, index=countries.index(ex.get('country', 'United States')))
    genres = st.multiselect("Genres / categories", bundle['genres'], default=ex.get('genres', []))
    desc = st.text_area("Plot description", value=ex.get('desc', ''), height=120,
                        placeholder="Paste or write the short plot summary…")
    with st.expander("More options"):
        e1, e2, e3 = st.columns(3)
        year = e1.number_input("Release year", 1925, 2030, ex.get('year', 2020))
        added_year = e2.number_input("Year added to Netflix", 2008, 2030, max(2021, ex.get('year', 2020)))
        month = e3.selectbox("Month added", list(range(1, 13)), index=0)
        e4, e5, e6 = st.columns(3)
        n_cast = e4.number_input("Number of cast members", 0, 50, 6)
        n_countries = e5.number_input("Number of countries", 1, 10, 1)
        has_director = e6.checkbox("Director listed", value=True)
    go = st.button("Predict maturity", type="primary", use_container_width=True)

with right:
    st.subheader("2. Prediction")
    if not go:
        st.info("Fill in the details and press **Predict maturity**.")
    elif not genres:
        st.warning("Please choose at least one genre (the most important input).")
    elif len(desc.split()) < 3:
        st.warning("Please add a short plot description (at least a few words).")
    else:
        row = pd.DataFrame([{
            'type': ctype, 'title': title, 'director': 'x' if has_director else None,
            'cast': ', '.join(['a'] * n_cast) if n_cast else None,
            'country': ', '.join([country] + ['x'] * (n_countries - 1)),
            'date_added': f"{dt.date(2000, month, 1).strftime('%B')} 1, {added_year}",
            'release_year': year, 'duration': duration, 'listed_in': ', '.join(genres), 'description': desc}])
        X = build_features(row)[FEATURES]
        proba = pd.Series(model.predict_proba(X)[0], index=model.classes_)[CLASSES]
        pred = proba.idxmax()
        st.markdown(f"<div style='padding:18px;border-radius:12px;background:{COLORS[pred]};color:white;text-align:center'>"
                    f"<div style='font-size:14px;opacity:.85'>Predicted maturity group</div>"
                    f"<div style='font-size:38px;font-weight:700'>{pred}</div>"
                    f"<div>{proba[pred]*100:.0f}% confidence</div></div>", unsafe_allow_html=True)
        st.markdown("&nbsp;")
        for c in CLASSES:
            st.write(f"**{c}** · {proba[c]*100:.1f}%"); st.progress(float(proba[c]))
        st.caption({'Kids': "Typical ratings: TV-Y, TV-Y7, TV-G, G",
                    'Teens': "Typical ratings: TV-PG, PG, TV-14, PG-13",
                    'Mature': "Typical ratings: TV-MA, R, NC-17"}[pred])
        if proba.sort_values().iloc[-1] - proba.sort_values().iloc[-2] < 0.15:
            st.warning("Close call: the top two classes are within 15 points. Teens vs Mature is the model's hardest distinction.")
