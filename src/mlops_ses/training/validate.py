import fire
import joblib
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, median_absolute_error
import mlflow


def validate(model_path, x_test_path, y_test_path):
    print(f"validate {model_path}")
    model = joblib.load(model_path)

    x_test = pd.read_csv(x_test_path)
    y_test = pd.read_csv(y_test_path)

    # S'assurer que y_test est un vecteur
    if y_test.shape[1] == 1:
        y_test = y_test.iloc[:, 0]

    y_pred = model.predict(x_test)

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    medae = median_absolute_error(y_test, y_pred)

    # Importance des features
    feature_names = x_test.columns.tolist()
    coefs = model.coef_
    # Si multi-output, coefs peut être 2D
    if hasattr(coefs, 'shape') and len(coefs.shape) > 1:
        coefs = coefs[0]
    feature_importance = {name: float(coef) for name, coef in zip(feature_names, coefs)}

    with mlflow.start_run():
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("medae", medae)
        mlflow.log_dict(feature_importance, "feature_importance.json")

    mlflow.sklearn.log_model(model, name="model", input_example=x_test, registered_model_name="model")


if __name__ == "__main__":
    fire.Fire(validate)