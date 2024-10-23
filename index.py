import streamlit as st
import pandas as pd
import os

st.set_page_config(
    page_title="Hello",
    page_icon="👋",
)

st.write("# Prediction Calories Burned by Gym Members! 👋")

st.sidebar.success("Select a demo above.")

dataset_path = st.file_uploader("Upload CSV", type=["csv"])
if dataset_path is None:
    dataset_path = os.path.join(os.path.dirname(__file__), 'gym_members_exercise_tracking.csv')

df = pd.read_csv(dataset_path)
st.session_state.df = df

df
