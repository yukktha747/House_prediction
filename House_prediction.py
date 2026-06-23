import streamlit as st
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression, Ridge, Lasso

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# ------------------------------------
# PAGE CONFIG
# ------------------------------------

st.set_page_config(
    page_title="House Price Prediction",
    layout="wide"
)

st.title("🏠 House Price Prediction Project")

# ------------------------------------
# LOAD DATA
# ------------------------------------

@st.cache_data
def load_data():
    return pd.read_csv("Housing.csv")

df = load_data()

# ------------------------------------
# SIDEBAR
# ------------------------------------

menu = st.sidebar.radio(
    "Navigation",
    [
        "Dataset",
        "EDA",
        "Model Training",
        "Feature Importance",
        "Prediction"
    ]
)

# ------------------------------------
# DATASET
# ------------------------------------

if menu == "Dataset":

    st.header("Dataset Overview")

    st.dataframe(df.head())

    st.subheader("Shape")
    st.write(df.shape)

    st.subheader("Missing Values")
    st.dataframe(df.isnull().sum())

    st.subheader("Statistics")
    st.dataframe(df.describe())

# ------------------------------------
# EDA
# ------------------------------------

elif menu == "EDA":

    st.header("Exploratory Data Analysis")

    st.subheader("Price Distribution")

    fig, ax = plt.subplots(figsize=(8,5))

    sns.histplot(df["price"], kde=True, ax=ax)

    st.pyplot(fig)

    st.subheader("Area vs Price")

    fig, ax = plt.subplots(figsize=(8,5))

    sns.scatterplot(
        x="area",
        y="price",
        data=df,
        ax=ax
    )

    st.pyplot(fig)

    # Encode for correlation

    temp_df = df.copy()

    yes_no_cols = [
        'mainroad',
        'guestroom',
        'basement',
        'hotwaterheating',
        'airconditioning',
        'prefarea'
    ]

    for col in yes_no_cols:
        temp_df[col] = temp_df[col].map({
            'yes':1,
            'no':0
        })

    temp_df = pd.get_dummies(
        temp_df,
        columns=['furnishingstatus'],
        drop_first=True
    )

    st.subheader("Correlation Heatmap")

    fig, ax = plt.subplots(figsize=(12,8))

    sns.heatmap(
        temp_df.corr(),
        cmap="coolwarm",
        ax=ax
    )

    st.pyplot(fig)

# ------------------------------------
# MODEL TRAINING
# ------------------------------------

elif menu == "Model Training":

    st.header("Model Training")

    df_model = df.copy()

    yes_no_cols = [
        'mainroad',
        'guestroom',
        'basement',
        'hotwaterheating',
        'airconditioning',
        'prefarea'
    ]

    for col in yes_no_cols:
        df_model[col] = df_model[col].map({
            'yes':1,
            'no':0
        })

    df_model = pd.get_dummies(
        df_model,
        columns=['furnishingstatus'],
        drop_first=True
    )

    X = df_model.drop("price", axis=1)
    y = df_model["price"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Linear

    linear = LinearRegression()

    linear.fit(
        X_train_scaled,
        y_train
    )

    linear_pred = linear.predict(
        X_test_scaled
    )

    # Ridge

    ridge = Ridge(alpha=1.0)

    ridge.fit(
        X_train_scaled,
        y_train
    )

    ridge_pred = ridge.predict(
        X_test_scaled
    )

    # Lasso

    lasso = Lasso(alpha=0.1)

    lasso.fit(
        X_train_scaled,
        y_train
    )

    lasso_pred = lasso.predict(
        X_test_scaled
    )

    comparison = pd.DataFrame({

        "Model":[
            "Linear Regression",
            "Ridge Regression",
            "Lasso Regression"
        ],

        "MAE":[
            mean_absolute_error(y_test,linear_pred),
            mean_absolute_error(y_test,ridge_pred),
            mean_absolute_error(y_test,lasso_pred)
        ],

        "RMSE":[
            np.sqrt(mean_squared_error(y_test,linear_pred)),
            np.sqrt(mean_squared_error(y_test,ridge_pred)),
            np.sqrt(mean_squared_error(y_test,lasso_pred))
        ],

        "R2":[
            r2_score(y_test,linear_pred),
            r2_score(y_test,ridge_pred),
            r2_score(y_test,lasso_pred)
        ]
    })

    st.dataframe(comparison)

# ------------------------------------
# FEATURE IMPORTANCE
# ------------------------------------

elif menu == "Feature Importance":

    st.header("Feature Importance")

    df_model = df.copy()

    yes_no_cols = [
        'mainroad',
        'guestroom',
        'basement',
        'hotwaterheating',
        'airconditioning',
        'prefarea'
    ]

    for col in yes_no_cols:
        df_model[col] = df_model[col].map({
            'yes':1,
            'no':0
        })

    df_model = pd.get_dummies(
        df_model,
        columns=['furnishingstatus'],
        drop_first=True
    )

    X = df_model.drop("price", axis=1)
    y = df_model["price"]

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    model = LinearRegression()

    model.fit(X_scaled,y)

    importance = pd.DataFrame({
        "Feature": X.columns,
        "Coefficient": model.coef_
    })

    st.dataframe(importance)

    st.bar_chart(
        importance.set_index("Feature")
    )

# ------------------------------------
# PREDICTION
# ------------------------------------

elif menu == "Prediction":

    st.header("🏠 Predict House Price")

    # --------------------
    # Train Model
    # --------------------

    df_model = df.copy()

    yes_no_cols = [
        'mainroad',
        'guestroom',
        'basement',
        'hotwaterheating',
        'airconditioning',
        'prefarea'
    ]

    for col in yes_no_cols:
        df_model[col] = df_model[col].map({
            'yes': 1,
            'no': 0
        })

    df_model = pd.get_dummies(
        df_model,
        columns=['furnishingstatus'],
        drop_first=True
    )

    X = df_model.drop("price", axis=1)
    y = df_model["price"]

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    model = LinearRegression()

    model.fit(X_scaled, y)

    # --------------------
    # User Inputs
    # --------------------

    area = st.number_input(
        "Area",
        min_value=1000,
        max_value=20000,
        value=5000
    )

    bedrooms = st.number_input(
        "Bedrooms",
        min_value=1,
        max_value=10,
        value=3
    )

    bathrooms = st.number_input(
        "Bathrooms",
        min_value=1,
        max_value=10,
        value=2
    )

    stories = st.number_input(
        "Stories",
        min_value=1,
        max_value=5,
        value=2
    )

    parking = st.number_input(
        "Parking",
        min_value=0,
        max_value=5,
        value=1
    )

    mainroad = st.selectbox(
        "Main Road",
        ["yes", "no"]
    )

    guestroom = st.selectbox(
        "Guest Room",
        ["yes", "no"]
    )

    basement = st.selectbox(
        "Basement",
        ["yes", "no"]
    )

    hotwaterheating = st.selectbox(
        "Hot Water Heating",
        ["yes", "no"]
    )

    airconditioning = st.selectbox(
        "Air Conditioning",
        ["yes", "no"]
    )

    prefarea = st.selectbox(
        "Preferred Area",
        ["yes", "no"]
    )

    furnishing = st.selectbox(
        "Furnishing Status",
        [
            "furnished",
            "semi-furnished",
            "unfurnished"
        ]
    )

    # --------------------
    # Prediction
    # --------------------

    if st.button("Predict Price"):

        input_data = pd.DataFrame({

            "area": [area],
            "bedrooms": [bedrooms],
            "bathrooms": [bathrooms],
            "stories": [stories],

            "mainroad": [
                1 if mainroad == "yes" else 0
            ],

            "guestroom": [
                1 if guestroom == "yes" else 0
            ],

            "basement": [
                1 if basement == "yes" else 0
            ],

            "hotwaterheating": [
                1 if hotwaterheating == "yes" else 0
            ],

            "airconditioning": [
                1 if airconditioning == "yes" else 0
            ],

            "parking": [parking],

            "prefarea": [
                1 if prefarea == "yes" else 0
            ],

            "furnishingstatus_semi-furnished": [
                1 if furnishing == "semi-furnished" else 0
            ],

            "furnishingstatus_unfurnished": [
                1 if furnishing == "unfurnished" else 0
            ]
        })

        input_scaled = scaler.transform(input_data)

        prediction = model.predict(input_scaled)

        st.success(
            f"Estimated House Price: ₹ {prediction[0]:,.0f}"
        )