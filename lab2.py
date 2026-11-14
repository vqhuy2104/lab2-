
# =====================================================
# ITA202 - LAB 2: DU LIEU VA TIEN XU LY
# =====================================================

import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes, load_digits
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


# =====================================================
# BAI 1: LAM SACH DU LIEU
# =====================================================

print("\n========== BAI 1: LAM SACH DU LIEU ==========")

# Tao du lieu co gia tri thieu va ngoai le
data = {
    "Diem": [20, 22, 24, 25, 26, 27, 28, 29, 30, 31, 32, np.nan, 200]
}

df = pd.DataFrame(data)

print("\nDu lieu ban dau:")
print(df)

# Dien gia tri thieu bang gia tri trung binh
mean_value = df["Diem"].mean()
df["Diem"] = df["Diem"].fillna(mean_value)

print("\nDu lieu sau khi dien gia tri thieu:")
print(df)

# Tinh trung binh va do lech chuan
mean_value = df["Diem"].mean()
std_value = df["Diem"].std()

# Xac dinh nguong ngoai le
lower_limit = mean_value - 3 * std_value
upper_limit = mean_value + 3 * std_value

print("\nNguong duoi:", lower_limit)
print("Nguong tren:", upper_limit)

# Xu ly gia tri ngoai le bang gia tri trung binh
outlier_condition = (
    (df["Diem"] < lower_limit) |
    (df["Diem"] > upper_limit)
)

df.loc[outlier_condition, "Diem"] = mean_value

print("\nDu lieu sau khi xu ly ngoai le:")
print(df)


# =====================================================
# BAI 2: CHUAN HOA DU LIEU (SCALING)
# =====================================================

print("\n========== BAI 2: CHUAN HOA DU LIEU ==========")

# Tao mang du lieu gom Tuoi va Thu nhap
X = np.array([
    [20, 10],
    [25, 20],
    [30, 30],
    [35, 40],
    [40, 50],
    [45, 60],
    [50, 80],
    [60, 100]
])

print("\nDu lieu ban dau:")
print(X)

# Su dung StandardScaler
standard_scaler = StandardScaler()
X_standard = standard_scaler.fit_transform(X)

print("\nKet qua StandardScaler:")
print(X_standard)

# Su dung MinMaxScaler
minmax_scaler = MinMaxScaler()
X_minmax = minmax_scaler.fit_transform(X)

print("\nKet qua MinMaxScaler:")
print(X_minmax)


# =====================================================
# BAI 3: CHIA TAP DU LIEU TRAIN/TEST
# =====================================================

print("\n========== BAI 3: TRAIN/TEST ==========")

# Tai du lieu diabetes
diabetes = load_diabetes()

X = diabetes.data
y = diabetes.target

print("\nKich thuoc du lieu goc:")
print("X:", X.shape)
print("y:", y.shape)

# Chia du lieu theo ty le 80% va 20%
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nKich thuoc sau khi chia:")
print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)


# =====================================================
# BAI 4: QUY TRINH ML HOAN CHINH VOI DIGITS
# =====================================================

print("\n========== BAI 4: PIPELINE VOI DIGITS ==========")

# Tai du lieu digits
digits = load_digits()

X = digits.data
y = digits.target

print("\nKich thuoc du lieu digits:")
print("X:", X.shape)
print("y:", y.shape)

# Chia du lieu thanh tap train va test
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nKich thuoc tap train va test:")
print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)

# Tao Pipeline gom StandardScaler va SVC
model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", SVC())
])

# Huan luyen mo hinh
model.fit(X_train, y_train)

# Du doan ket qua
y_pred = model.predict(X_test)

# Tinh do chinh xac
accuracy = accuracy_score(y_test, y_pred)

print("\nDo chinh xac cua mo hinh:")
print(accuracy)

print("\nDo chinh xac (%):")
print(accuracy * 100)


# =====================================================
# KET THUC LAB 2
# =====================================================

print("\n========== HOAN THANH LAB 2 ==========")
