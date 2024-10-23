import streamlit as st
import pandas as pd

page_title = "Prediction"
st.set_page_config(page_title=page_title, page_icon="🎉")
st.sidebar.header(page_title)
st.markdown(f'# {page_title}')

df = st.session_state.df

from openai import OpenAI
client = OpenAI(api_key = "API_KEY")

completion = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": "Write a haiku about recursion in programming."
        }
    ]
)

def generate_gpt_recommendation(predicted_calories, goal_calories=500, weight=None, experience_level=None):
    """
    Generates personalized fitness recommendations using GPT API (gpt-4o-mini).

    Args:
    - predicted_calories (float): Predicted calories burned.
    - goal_calories (float): Target calories to burn.
    - weight (float): User's weight (optional).
    - experience_level (str): User's fitness experience level (optional).

    Returns:
    - recommendation (str): The GPT-generated recommendation.
    """

    # Construct the user message for GPT
    user_message = f"""
    I have predicted that a user has burned {predicted_calories} calories in their workout today.
    Their goal is to burn {goal_calories} calories.
    """

    # Optionally, add more context to the user message
    if weight:
        user_message += f"\nThe user weighs {weight} kg."
    if experience_level:
        user_message += f"\nThe user's fitness experience level is {experience_level}."

    user_message += "\nPlease provide a personalized fitness recommendation based on this information."

    # Make the API call to GPT
    completion = client.chat.completions.create(
        model="gpt-4o-mini",  # Use gpt-4o-mini as specified
        messages=[
            {"role": "user", "content": user_message}
        ]
    )

    # Extract and return the recommendation
    # Use .content instead of ['content'] to access the message content
    recommendation = completion.choices[0].message.content
    return recommendation

models = st.session_state.models

gender_col = st.session_state.gender_col
gender_mapping_name2id = st.session_state.gender_mapping_name2id
new_gender_col = st.session_state.new_gender_col

label_col = st.session_state.label_col

sample_df = st.session_state.sample_df

with st.form("predict_form"):
    
    model_name = st.selectbox("Choose model", models[0].keys())

    sample_df = st.data_editor(sample_df)
    sample_row = sample_df.iloc[0]

    submitted = st.form_submit_button("Start predict")
    if submitted:
        
        gender_id = int(sample_row[new_gender_col])
        selected_model = models[gender_id][model_name]
        
        predicted_calories = selected_model.predict([sample_row])
        st.markdown(f'**{label_col}**: {predicted_calories[0]}')

        goal_calories = 500
        weight = 75
        experience_level = "intermediate"  # Can be beginner, intermediate, or advanced

        recommendation = generate_gpt_recommendation(predicted_calories, goal_calories, weight, experience_level)
        st.markdown(recommendation)
