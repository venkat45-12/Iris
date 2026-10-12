"""Predict an iris species from the command line.

Example: python predict.py 5.1 3.5 1.4 0.2
"""
import argparse
import json
import joblib
import pandas as pd


def main():
    p = argparse.ArgumentParser(description="Iris species predictor")
    p.add_argument("sepal_length", type=float)
    p.add_argument("sepal_width", type=float)
    p.add_argument("petal_length", type=float)
    p.add_argument("petal_width", type=float)
    args = p.parse_args()

    model = joblib.load("models/iris_model.joblib")
    meta = json.load(open("models/metadata.json"))
    sample = pd.DataFrame([[args.sepal_length, args.sepal_width,
                            args.petal_length, args.petal_width]],
                          columns=meta["features"])
    pred = model.predict(sample)[0]
    proba = model.predict_proba(sample)[0]
    print(f"Predicted species: {meta['classes'][pred]}")
    for cls, pr in zip(meta["classes"], proba):
        print(f"  {cls:12s} {pr:.1%}")


if __name__ == "__main__":
    main()
