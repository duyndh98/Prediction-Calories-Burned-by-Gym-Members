import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.base import clone

page_title = "Training"
st.set_page_config(page_title=page_title, layout="wide", page_icon="🌍")
st.sidebar.header(page_title)
st.markdown(f'# {page_title}')

df = st.session_state.df

st.markdown("## label")
label_col = st.selectbox("", ("Calories_Burned"))
st.session_state.label_col = label_col

gender_col = "Gender"
st.session_state.gender_col = gender_col
gender_names = st.session_state.df[gender_col].unique()

st.markdown("## features")
all_feature_cols = [col for col in df.columns if col != label_col and col != gender_col]
feature_cols = st.multiselect("", all_feature_cols, default=all_feature_cols)
if st.checkbox("Select all features", value=True):
    feature_cols = all_feature_cols
feature_cols.append(gender_col)

df = st.session_state.df[feature_cols + [label_col]]

if "Workout_Type" in df.columns:
    df = pd.get_dummies(df, columns=["Workout_Type"], drop_first=True)

df = pd.get_dummies(df, columns=[gender_col], drop_first=True)
df

sample_df = df[:1].copy(deep=True)
sample_df = sample_df.drop(label_col, axis=1)
st.session_state.sample_df = sample_df

new_gender_col = list(df.columns.difference(feature_cols + [label_col]))[0]
st.session_state.new_gender_col = new_gender_col
gender_ids = df[new_gender_col].unique()

feature_cols = [name for name in df.columns if name != label_col]

gender_mapping_id2name = {key: value for key, value in zip(gender_ids, gender_names)}
st.session_state.gender_mapping_name2id = {key: value for key, value in zip(gender_names, gender_ids)}

test_size = st.sidebar.slider("Test size", 0.1, 0.9, 0.2, 0.1)
n_iter = st.sidebar.number_input("Number of iter", value=5)

models_dict = {
    "LinearRegression": LinearRegression(),
    "RandomForestRegressor": RandomizedSearchCV(
        estimator=RandomForestRegressor(random_state=42),
        param_distributions={
            'n_estimators': np.arange(50, 300, 50),
            'max_depth': [None, 10, 20, 30, 50],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': [None, 'sqrt', 'log2']
        },
        n_iter=n_iter,  # Number of combinations to try
        cv=5,  # 5-fold cross-validation
        scoring='neg_mean_squared_error',
        n_jobs=-1,  # Use all available processors
        random_state=42
    ),
    "XGBRegressor": RandomizedSearchCV(
        estimator=XGBRegressor(random_state=42),
        param_distributions={
            'n_estimators': np.arange(50, 500, 50),  # Number of trees
            'learning_rate': [0.01, 0.05, 0.1, 0.2],  # Step size
            'max_depth': [3, 5, 7, 10],  # Tree depth
            'subsample': [0.6, 0.8, 1.0],  # Fraction of samples used per tree
            'colsample_bytree': [0.6, 0.8, 1.0],  # Fraction of features used per tree
            'min_child_weight': [1, 5, 10],  # Minimum weight in leaf nodes
            'gamma': [0, 0.1, 0.5, 1],  # Regularization term
        },
        n_iter=n_iter,  # Number of random combinations to try
        cv=5,  # 5-fold cross-validation
        scoring='neg_mean_squared_error',  # Minimize MSE
        n_jobs=-1,  # Use all CPU cores
        random_state=42
    ),
    "LGBMRegressor": RandomizedSearchCV(
        estimator=LGBMRegressor(random_state=42),
        param_distributions={
            'n_estimators': np.arange(50, 500, 50),  # Number of boosting rounds
            'learning_rate': [0.01, 0.05, 0.1, 0.2],  # Step size
            'max_depth': [-1, 5, 10, 15, 20],  # Maximum tree depth (-1 means no limit)
            'num_leaves': [31, 50, 100, 200],  # Maximum number of leaves
            'min_child_samples': [5, 10, 20, 30],  # Minimum number of samples in a leaf
            'subsample': [0.6, 0.8, 1.0],  # Fraction of samples to be used for each tree
            'colsample_bytree': [0.6, 0.8, 1.0],  # Fraction of features to be used for each tree
        },
        n_iter=n_iter,  # Number of random combinations to try
        cv=5,  # 5-fold cross-validation
        scoring='neg_mean_squared_error',  # Minimize MSE
        n_jobs=-1,  # Use all available CPU cores
        random_state=42
    )
}

has_trained = 'results' in st.session_state
models = st.session_state.models if has_trained else {}
results = st.session_state.results if has_trained else {}

for gender_id in gender_ids:
    gender_name = gender_mapping_id2name[gender_id]
    st.markdown(f"## {gender_name}")
    
    df_g = df[df[new_gender_col] == gender_id]    
    X_g = df_g[feature_cols]
    y_g = df_g[label_col]

    X_train, X_test, y_train, y_test = train_test_split(X_g, y_g, test_size=test_size, random_state=42)

    gender_models = models[gender_id] if has_trained else {}
    gender_results = results[gender_id] if has_trained else { "MAE": {}, "MSE": {}, "R_2": {} }

    fig, axes = plt.subplots(1, len(models_dict), figsize=(10, 5))
    
    for model_id, model_name in enumerate(models_dict):
        
        model = None
        if has_trained:
            model = gender_models[model_name]
            y_pred = model.predict(X_test)
        else:
            model = clone(models_dict[model_name])
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            gender_models[model_name] = model

            gender_results["MAE"][model_name] = mean_absolute_error(y_test, y_pred)
            gender_results["MSE"][model_name] = mean_squared_error(y_test, y_pred)
            gender_results["R_2"][model_name] = r2_score(y_test, y_pred)

        axes[model_id].scatter(x=y_test, y=y_pred, s=8)        
        axes[model_id].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')
        axes[model_id].set_title(f'{model_name}')
        axes[model_id].set_xlabel('Actual Calories Burned')
        axes[model_id].set_ylabel('Predicted Calories Burned')

    st.markdown("### MAE")
    gender_results["MAE"]

    st.markdown("### MSE")
    gender_results["MSE"]
    
    st.markdown("### R^2")
    gender_results["R_2"]

    models[gender_id] = gender_models
    results[gender_id] = gender_results

    plt.tight_layout()
    st.pyplot(fig)

if not has_trained:
    st.session_state.models = models
    st.session_state.results = results


# if 'male_models' not in st.session_state:
#     st.session_state.male_models = male_models

# if 'female_models' not in st.session_state:
#     st.session_state.female_models = female_models