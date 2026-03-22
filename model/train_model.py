import os
import joblib
import numpy as np
from sklearn.datasets        import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing   import StandardScaler
from sklearn.linear_model    import LogisticRegression
from sklearn.tree            import DecisionTreeClassifier
from sklearn.ensemble        import RandomForestClassifier
from sklearn.svm             import SVC
from sklearn.metrics         import accuracy_score, classification_report

os.makedirs("outputs/models", exist_ok=True)


def train():
    print("\n" + "═" * 52)
    print("  IRIS FLOWER CLASSIFIER — MODEL TRAINING")
    print("  Syntecxhub ML Internship | Week 2")
    print("═" * 52)

    iris     = load_iris()
    X, y     = iris.data, iris.target
    names    = iris.target_names

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler         = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)
    X_scaled_full  = scaler.transform(X)

    print(f"\n  Dataset  : {X.shape[0]} samples, {X.shape[1]} features")
    print(f"  Train    : {X_train.shape[0]} | Test: {X_test.shape[0]}")
    print(f"  Classes  : {list(names)}\n")

    models = {}

    lr = LogisticRegression(max_iter=200, random_state=42)
    lr.fit(X_train_scaled, y_train)
    lr_acc = accuracy_score(y_test, lr.predict(X_test_scaled))
    lr_cv  = cross_val_score(lr, X_scaled_full, y, cv=5).mean()
    models["logistic_regression"] = lr

    dt = DecisionTreeClassifier(max_depth=4, random_state=42)
    dt.fit(X_train, y_train)
    dt_acc = accuracy_score(y_test, dt.predict(X_test))
    dt_cv  = cross_val_score(dt, X, y, cv=5).mean()
    models["decision_tree"] = dt

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    rf_acc = accuracy_score(y_test, rf.predict(X_test))
    rf_cv  = cross_val_score(rf, X, y, cv=5).mean()
    models["random_forest"] = rf

    svm = SVC(kernel="rbf", probability=True, random_state=42)
    svm.fit(X_train_scaled, y_train)
    svm_acc = accuracy_score(y_test, svm.predict(X_test_scaled))
    svm_cv  = cross_val_score(svm, X_scaled_full, y, cv=5).mean()
    models["svm"] = svm

    print(f"  {'Model':<22} {'Test Acc':>10} {'CV Acc (5-fold)':>16}")
    print("  " + "─" * 50)
    results = [
        ("Logistic Regression", lr_acc,  lr_cv),
        ("Decision Tree",       dt_acc,  dt_cv),
        ("Random Forest",       rf_acc,  rf_cv),
        ("SVM",                 svm_acc, svm_cv),
    ]
    for name, acc, cv in results:
        print(f"  {name:<22} {acc*100:>9.2f}% {cv*100:>15.2f}%")

    best = max(results, key=lambda x: x[1])
    print(f"\n  ✔ Best model : {best[0]} ({best[1]*100:.2f}%)")

    print(f"\n  Classification Report ({best[0]}):")
    if best[0] == "Logistic Regression":
        preds = lr.predict(X_test_scaled)
    elif best[0] == "SVM":
        preds = svm.predict(X_test_scaled)
    elif best[0] == "Random Forest":
        preds = rf.predict(X_test)
    else:
        preds = dt.predict(X_test)
    print(classification_report(y_test, preds, target_names=names))

    for model_name, model_obj in models.items():
        joblib.dump(model_obj, f"outputs/models/{model_name}.pkl")
    joblib.dump(scaler, "outputs/models/scaler.pkl")

    importances = rf.feature_importances_.tolist()
    joblib.dump({
        "feature_names":  list(iris.feature_names),
        "importances":    importances,
        "target_names":   list(names),
    }, "outputs/models/metadata.pkl")

    print("  ✔ All models saved → outputs/models/")
    print("  ✔ Metadata saved   → outputs/models/metadata.pkl")
    print("\n" + "═" * 52)
    print("  Training complete. Run: python app.py")
    print("═" * 52 + "\n")


if __name__ == "__main__":
    train()
