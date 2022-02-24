from pathlib import Path

data_dir = Path(__file__).resolve().parents[3] / "data"

x_train = data_dir / "Xtr.csv"
x_val = data_dir / "Xval.csv"
y_train = data_dir / "Ytr.csv"
y_val = data_dir / "Yval.csv"
x_test = data_dir / "Xte.csv"
y_test = data_dir / "Yte.csv"

x_downloaded = [x_train, x_test]
y_downloaded = [y_train]
data_files = [x_train, x_val, x_test, y_train, y_val, y_test]
