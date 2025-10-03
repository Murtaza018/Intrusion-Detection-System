import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

# Load column names from file
with open("../Datasets/KDD/KDDNames.txt", "r") as f:
    col_names = f.read().strip().split(",")

col_names = [c.strip() for c in col_names]

# Load datasets
train_path = "../Datasets/KDD/KDDTrain.txt"
test_path = "../Datasets/KDD/KDDTest.txt"

df_train = pd.read_csv(train_path, names=col_names)
df_test = pd.read_csv(test_path, names=col_names)

# Drop difficulty_level
df_train.drop("difficulty_level", axis=1, inplace=True)
df_test.drop("difficulty_level", axis=1, inplace=True)

#  Convert labels to binary (normal=0, attack=1)
df_train["label"] = df_train["label"].apply(lambda x: 0 if x == "normal" else 1)
df_test["label"] = df_test["label"].apply(lambda x: 0 if x == "normal" else 1)

# Split features/labels
X_train = df_train.drop("label", axis=1)
y_train = df_train["label"]
X_test = df_test.drop("label", axis=1)
y_test = df_test["label"]

# Identify categorical and numeric columns
categorical_cols = ["protocol_type", "service", "flag"]
numeric_cols = [col for col in X_train.columns if col not in categorical_cols]

# Preprocessing pipeline
numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown="ignore")

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_cols),
        ("cat", categorical_transformer, categorical_cols)
    ]
)

# Fit on training and transform train/test
pipeline = Pipeline(steps=[("preprocessor", preprocessor)])

X_train_processed = pipeline.fit_transform(X_train)
X_test_processed = pipeline.transform(X_test)

#  Train-validation split
X_train_final, X_val, y_train_final, y_val = train_test_split(
    X_train_processed, y_train, test_size=0.15, random_state=42, stratify=y_train
)

#  Save processed data + pipeline for reuse
np.save("X_train.npy", X_train_final.toarray() if hasattr(X_train_final, "toarray") else X_train_final)
np.save("X_val.npy", X_val.toarray() if hasattr(X_val, "toarray") else X_val)
np.save("y_train.npy", y_train_final)
np.save("y_val.npy", y_val)
np.save("X_test.npy", X_test_processed.toarray() if hasattr(X_test_processed, "toarray") else X_test_processed)
np.save("y_test.npy", y_test)

joblib.dump(pipeline, "preprocessing_pipeline.pkl")

print("Preprocessing complete! Data saved.")
