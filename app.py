"""Streamlit web app. Run: streamlit run app.py"""
import json
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Iris Classifier", page_icon="🌸")
st.title("🌸 Iris Flower Classifier")

@st.cache_resource
def load():
    return joblib.load("models/iris_model.joblib"), json.load(open("models/metadata.json"))

model, meta = load()
st.caption(f"Model: {meta['best_model']} | Test accuracy: {meta['test_accuracy']:.1%}")

col1, col2 = st.columns(2)
sl = col1.slider("Sepal length (cm)", 4.0, 8.0, 5.8, 0.1)
sw = col1.slider("Sepal width (cm)", 2.0, 4.5, 3.0, 0.1)
pl = col2.slider("Petal length (cm)", 1.0, 7.0, 4.3, 0.1)
pw = col2.slider("Petal width (cm)", 0.1, 2.5, 1.3, 0.1)

sample = pd.DataFrame([[sl, sw, pl, pw]], columns=meta["features"])
pred = model.predict(sample)[0]
proba = model.predict_proba(sample)[0]

st.subheader(f"Prediction: {meta['classes'][pred].capitalize()}")
st.bar_chart(pd.DataFrame({"probability": proba}, index=meta["classes"]))
