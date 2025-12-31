import joblib
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
from src.utils import print_header

def train_ann(X_train, X_test, y_train, y_test, params, save_path):
    print_header("traning mlp")

    if "hidden_layers" in params:
        params["hidden_layer_sizes"] = tuple(params["hidden_layers"])
        del params["hidden_layers"]

    ann = MLPClassifier(**params)
    ann.fit(X_train, y_train)
    preds = ann.predict(X_test)

    print("MLP Accuracy:", accuracy_score(y_test, preds))
    print("Report:\n", classification_report(y_test, preds))

    joblib.dump(ann, save_path)
    print(f"MLP saved to {save_path}")