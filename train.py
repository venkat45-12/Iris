"""Train and compare several classifiers on the Iris dataset."""
import json
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

RANDOM_STATE = 42


def main():
    # 1. Load data
    iris = load_iris(as_frame=True)
    X, y = iris.data, iris.target
    class_names = list(iris.target_names)
    print(f"Samples: {X.shape[0]}, Features: {X.shape[1]}, Classes: {class_names}")

    # 2. Explore: pairplot
    df = iris.frame.copy()
    df["species"] = df["target"].map(dict(enumerate(class_names)))
    sns.pairplot(df.drop(columns="target"), hue="species")
    plt.savefig("outputs/pairplot.png", dpi=120, bbox_inches="tight")
    plt.close()

    # 3. Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    # 4. Candidate models
    models = {
        "Logistic Regression": LogisticRegression(max_iter=500),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "SVM": SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE),
    }
    pipelines = {n: Pipeline([("scaler", StandardScaler()), ("clf", m)]) for n, m in models.items()}

    # 5. Cross-validate and pick the best
    results = {}
    for name, pipe in pipelines.items():
        scores = cross_val_score(pipe, X_train, y_train, cv=5)
        results[name] = scores.mean()
        print(f"{name:20s} CV accuracy: {scores.mean():.3f} (+/- {scores.std():.3f})")
    best_name = max(results, key=results.get)
    best = pipelines[best_name]
    print(f"\nBest model: {best_name}")

    # 6. Fit and evaluate on held-out test set
    best.fit(X_train, y_train)
    preds = best.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"Test accuracy: {acc:.3f}\n")
    print(classification_report(y_test, preds, target_names=class_names))

    cm = confusion_matrix(y_test, preds)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.title(f"{best_name} - Confusion Matrix")
    plt.savefig("outputs/confusion_matrix.png", dpi=120, bbox_inches="tight")
    plt.close()

    # 7. Save model and metadata
    joblib.dump(best, "models/iris_model.joblib")
    with open("models/metadata.json", "w") as f:
        json.dump({"best_model": best_name, "test_accuracy": acc,
                   "classes": class_names, "features": list(X.columns)}, f, indent=2)
    print("Saved model to models/iris_model.joblib")


if __name__ == "__main__":
    main()
