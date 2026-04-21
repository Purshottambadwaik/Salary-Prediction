
import streamlit as st
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load the saved model pipeline
# The pipeline includes the preprocessor and the Random Forest Regressor
model_pipeline = joblib.load('best_model.pkl')

# Get unique values for select boxes from the training data (assuming df is available from notebook)
# This is a simplification; in a real deployment, these lists would be pre-saved or fetched.
# For 'Gender' and 'Education Level', we can use the unique values observed during training.
# For 'Job Title', using a text input is more flexible due to high cardinality.

# Note: In a production environment, you would save these unique values along with the model
# For this demonstration, we'll use a snapshot or assume they are known.

# Access the original dataframe (df) used for training to get categories
# This assumes the original 'df' is available, or you would load it here.
# For this Colab environment, 'df' is in the kernel state.
# If running as a standalone app, you'd need to load the original data or define categories explicitly.

# Let's re-define the features and their categories based on the original data
# This is crucial for the OneHotEncoder within the loaded pipeline

# Define the categorical features and their expected categories based on the training data
# This is a crucial step to ensure the Streamlit app uses the same encoding as the trained model.
# We need to extract these from the fitted OneHotEncoder within the pipeline.
# Let's inspect the fitted preprocessor from the loaded model_pipeline

# Function to extract categories from the fitted OneHotEncoder
def get_categories_from_ohe(pipeline, categorical_feature_names):
    ohe = pipeline.named_steps['preprocessor'].named_transformers_['cat']
    all_categories = ohe.categories_
    # Map categories back to original feature names
    # The order of categories_ corresponds to the order of categorical_features
    categories_map = {name: cat.tolist() for name, cat in zip(categorical_feature_names, all_categories)}
    return categories_map

categorical_feature_names = ['Gender', 'Education Level', 'Job Title']
# This call would only work if the model_pipeline was re-fit or we explicitly passed the preprocessor used for fitting
# For this demonstration, we will manually define some common options.

gender_options = ['Male', 'Female', 'Other'] # Added 'Other' for robustness
education_level_options = ['High School', 'Bachelor\'s Degree', 'Master\'s Degree', 'PhD'] # Standardizing based on observed data

# Streamlit App Title
st.title('Salary Prediction App')
st.write('Enter the details below to predict the salary.')

# User Inputs
age = st.slider('Age', 18, 65, 30)
gender = st.selectbox('Gender', gender_options)
education_level = st.selectbox('Education Level', education_level_options)
job_title = st.text_input('Job Title', 'Software Engineer')
years_of_experience = st.slider('Years of Experience', 0.0, 40.0, 5.0)

# Create a DataFrame from user inputs
input_data = pd.DataFrame({
    'Age': [age],
    'Gender': [gender],
    'Education Level': [education_level],
    'Job Title': [job_title],
    'Years of Experience': [years_of_experience]
})

# Predict button
if st.button('Predict Salary'):
    try:
        # Make prediction using the loaded pipeline
        predicted_salary = model_pipeline.predict(input_data)[0]
        st.success(f'The predicted salary is: ${predicted_salary:,.2f}')
    except Exception as e:
        st.error(f'An error occurred during prediction: {e}')
        st.write('Please ensure all input fields are correctly filled and valid.')

st.write("--- Developed for demonstration --- ")
