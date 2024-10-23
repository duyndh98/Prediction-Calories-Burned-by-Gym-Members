import streamlit as st
import pandas as pd
import time
import numpy as np
import os
import io
import matplotlib.pyplot as plt
import seaborn as sns

page_title = "Analysis"
st.set_page_config(page_title=page_title, page_icon="📈")
st.sidebar.header(page_title)
st.markdown(f'# {page_title}')

df = st.session_state.df.copy(deep=True)

st.markdown("## info")
buffer = io.StringIO()
df.info(buf=buffer)
st.text(buffer.getvalue())

st.markdown("## isnull")
st.write(df.isnull().sum())

st.markdown("## columns")
st.write(df.columns)

st.markdown("## shape")
st.write(df.shape)

st.markdown("## duplicated")
st.write(df.duplicated().sum())

# --------------------------------

st.markdown("## Count Plot")

df0=df.copy()
categorical_columns=df.select_dtypes(include=['object']).columns
fig, ax = plt.subplots(figsize=(6,8))

# Loop through categorical columns and plot
for i, col in enumerate(categorical_columns, 1):
    plt.subplot(len(categorical_columns), 1, i)
    ax = sns.countplot(data=df, x=col, order=df[col].value_counts().index)
    plt.title(f'Count Plot for {col}')
    plt.xticks(rotation=45, ha='right')

    # Add counts on top of bars
    for p in ax.containers:
        ax.bar_label(p)

plt.tight_layout()
st.pyplot(fig)

# --------------------------------

st.markdown("## describe")

Numerical_columns=df.select_dtypes(include=['float64','int64']).columns
st.write(df[Numerical_columns].describe().T)

# --------------------------------

st.markdown("## Boxplots of Age Distribution")

# Makeup
sns.set_style('whitegrid')

# twins for suplot

fig, (ax1,ax2) = plt.subplots(2,1, figsize =(12,10))

# Histogram of AGE
sns.histplot(data=df, x= 'Age', kde= True , bins=20, color='teal', ax=ax1)
ax1.set_title('Distribution of GYM Member Ages')

ax1.set_xlabel('Age')
ax1.set_ylabel('Count')

# Boxplot of age by Exprience level
sns.boxplot(data=df, x='Experience_Level', y='Age', ax=ax2)
ax2.set_title('Age Distribution by Experience Level')
ax2.set_xlabel('Experience Level')
ax2.set_ylabel('Age')
plt.tight_layout()
st.pyplot(fig)

# --------------------------------

st.markdown("## BarPlots by Gender")

fig,(ax1,ax2,ax3)= plt.subplots(3,1,figsize=(10,15))

# BarPlot Gender vs calories burned

sns.barplot(data=df, x='Gender', y='Calories_Burned', palette='viridis', ax=ax1)
ax1.set_title('Calories Burned by Gender', fontsize=14)
ax1.set_xlabel('Gender', fontsize=12)
ax1.set_ylabel('Calories Burned', fontsize=12)

# Barplot Gender vs max BPM
sns.barplot(data=df, x='Gender', y='Max_BPM', palette='viridis', ax=ax2)
ax2.set_title('Maximum Heart Rate (Max_BPM) by Gender', fontsize=14)
ax2.set_xlabel('Gender', fontsize=12)
ax2.set_ylabel('Max BPM', fontsize=12)

# 3. Countplot: Gender vs. Workout Type
sns.countplot(data=df, x='Workout_Type', hue='Gender', palette='viridis', ax=ax3)
ax3.legend(loc='upper right')
ax3.set_title('Workout Type Distribution by Gender', fontsize=14)
ax3.set_xlabel('Workout Type', fontsize=12)
ax3.set_ylabel('Count', fontsize=12)

st.pyplot(fig)

# --------------------------------

st.markdown("## Height vs Weight by Gender")

fig = plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='Height (m)', y='Weight (kg)', hue='Gender')
plt.title('Height vs Weight by Gender')
st.pyplot(fig)

# --------------------------------

st.markdown("## Box Plot of Weight by Age Group")

# For age groups
df['Age_Group'] = pd.cut(df['Age'], bins=[0, 20, 40, 60, 100], labels=['0-20', '21-40', '41-60', '60+'])

fig = plt.figure(figsize=(12, 6))
sns.boxplot(data=df, x='Age_Group', y='Weight (kg)', palette='viridis')
plt.title('Box Plot of Weight by Age Group', fontsize=16)
plt.xlabel('Age Group', fontsize=14)
plt.ylabel('Weight (kg)', fontsize=14)
st.pyplot(fig)

# --------------------------------

st.markdown("## Violin Plot of Heigh by Age Group")

fig = plt.figure(figsize=(12, 6))
sns.violinplot(data=df, x='Age_Group', y='Height (m)', palette='viridis')
plt.title('Violin Plot of Heigh by Age Group', fontsize=16)
plt.xlabel('Age Group', fontsize=14)
plt.ylabel('Height (m)', fontsize=14)
st.pyplot(fig)

# --------------------------------

st.markdown("## Max BPM by Workout Type")

fig = plt.figure(figsize =(12,8))

sns.boxplot(data=df, x='Workout_Type', y='Max_BPM')
plt.title('Max BPM by Workout Type')
plt.xlabel('Workout Type')
plt.ylabel('Max BPM')
st.pyplot(fig)

# --------------------------------

st.markdown("## Avg BPM by Gender")

fig = plt.figure(figsize=(12, 8))
sns.boxplot(data=df, x='Gender', y='Avg_BPM')
plt.title('Avg BPM by Gender')
plt.xlabel('Gender')
plt.ylabel('Avg BPM')
st.pyplot(fig)

# --------------------------------

st.markdown("## Pairplot of BPM Features Colored by Gender")

fig = sns.pairplot(df[['Max_BPM', 'Avg_BPM', 'Resting_BPM', 'Gender']], hue='Gender')
plt.suptitle('Pairplot of BPM Features Colored by Gender', y=1.02)
st.pyplot(fig)

# --------------------------------

st.markdown("## Calories Burned vs. Session Duration")

# Scatter Plot: Calories_Burned vs. Session_Duration
fig = plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='Session_Duration (hours)', y='Calories_Burned', hue='Gender', palette='viridis')
plt.title('Calories Burned vs. Session Duration')
plt.xlabel('Session Duration (hours)')
plt.ylabel('Calories Burned')
plt.legend(title='Workout Type', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
st.pyplot(fig)

# --------------------------------

st.markdown("## Average Calories Burned by Workout Type")

# Barplot: Average Calories_Burned by Workout_Type
fig = plt.figure(figsize=(8, 6))
sns.barplot(data=df, x='Workout_Type', y='Calories_Burned', estimator=np.mean, palette='viridis')
plt.title('Average Calories Burned by Workout Type')
plt.xlabel('Workout Type')
plt.ylabel('Average Calories Burned')
plt.xticks(rotation=45)
plt.tight_layout()
st.pyplot(fig)

# --------------------------------

st.markdown("## Scatter Plot of Fat Percentage vs. BMI")

# Scatter Plot: Fat_Percentage vs. BMI
fig = plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='BMI', y='Fat_Percentage', hue='Gender')
plt.title('Scatter Plot of Fat Percentage vs. BMI')
plt.xlabel('BMI')
plt.ylabel('Fat Percentage (%)')
plt.legend(title='Legend', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
st.pyplot(fig)

# --------------------------------

st.markdown("## Hexbin Plot of Water Intake vs. Calories Burned")

fig = plt.figure(figsize=(8, 5))
plt.hexbin(df['Calories_Burned'], df['Water_Intake (liters)'], gridsize=30, cmap='Blues', mincnt=1)
plt.colorbar(label='Counts')
plt.title('Hexbin Plot of Water Intake vs. Calories Burned')
plt.xlabel('Calories Burned')
plt.ylabel('Water Intake (liters)')
plt.tight_layout()
st.pyplot(fig)

# --------------------------------

st.markdown("## Workout Frequency vs. BMI Colored by Calories Burned")

fig = plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='Workout_Frequency (days/week)', y='BMI', size='Calories_Burned', hue='Calories_Burned',  palette='viridis')
plt.title('Workout Frequency vs. BMI Colored by Calories Burned')
plt.xlabel('Workout Frequency (days/week)')
plt.ylabel('BMI')
plt.legend(title='Calories Burned', loc='upper left', bbox_to_anchor=(1, 1))
plt.tight_layout()
st.pyplot(fig)

# --------------------------------

st.markdown("## Calories Burned by Experience Level")

fig = plt.figure(figsize=(12, 6))
sns.boxplot(data=df, x='Experience_Level', y='Calories_Burned', palette='viridis')
plt.title('Calories Burned by Experience Level')
plt.xlabel('Experience Level')
plt.ylabel('Calories Burned')
plt.tight_layout()
st.pyplot(fig)

# --------------------------------

st.markdown("## Workout Frequency by Experience Level")

fig = plt.figure(figsize=(12, 6))
sns.boxplot(data=df, x='Experience_Level', y='Workout_Frequency (days/week)', palette='viridis')
plt.title('Workout Frequency by Experience Level')
plt.xlabel('Experience Level')
plt.ylabel('Workout Frequency (days/week)')
plt.tight_layout()
st.pyplot(fig)

# --------------------------------

st.markdown("## Scatter Plot of BMI vs. Calories Burned")

fig = plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='BMI', y='Calories_Burned', hue='Experience_Level', palette='Set2', alpha=0.7)
plt.title('Scatter Plot of BMI vs. Calories Burned')
plt.xlabel('BMI')
plt.ylabel('Calories Burned')
plt.legend(title='Experience Level')
plt.tight_layout()
st.pyplot(fig)

# --------------------------------

st.markdown("## Feature Correlation Heatmap")

numerical_df = df.select_dtypes(include=[np.number])

# Calculate the correlation matrix
corr_matrix = numerical_df.corr()

# Create a heatmap
fig = plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
plt.title('Feature Correlation Heatmap')
st.pyplot(fig)

