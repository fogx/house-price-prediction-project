from pathlib import Path
import pandas as pd


def get_project_root() -> Path:
    return Path(__file__).absolute().parent.parent.resolve()


def get_log_dir() -> Path:
    return get_project_root() / "logs"


def create_submission(model, x_test, name: str = "result"):
    df_predictions = pd.DataFrame(x_test.index, columns=["Id"])
    df_predictions["Id"] = df_predictions["Id"] + 1  # the index started at 0, while Id starts at 1, so +1
    df_predictions.set_index("Id", inplace=True)
    df_predictions["SalePrice"] = model.predict(x_test)
    df_predictions.to_csv(get_project_root() / "data" / f"{name}.csv")
