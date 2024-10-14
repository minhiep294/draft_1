import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from openai import OpenAI

# Set up the OpenAI Document Question Answering section
st.title("📄 Document Question Answering & Exploratory Data Analysis App")

# Description for the OpenAI part
st.subheader("OpenAI GPT-powered Document QA")
st.write(
    "Upload a document and ask a question about it – GPT will answer! "
    "To use this feature, you need to provide an OpenAI API key, which you can get [here](https://platform.openai.com/account/api-keys)."
)

# Ask user for their OpenAI API key
openai_api_key = st.text_input("OpenAI API Key", type="password")

if openai_api_key:
    # Create an OpenAI client
    client = OpenAI(api_key=openai_api_key)

    # Let the user upload a file via `st.file_uploader`
    uploaded_file = st.file_uploader("Upload a document (.txt or .md)", type=("txt", "md"))

    # Ask the user for a question about the document
    question = st.text_area("Now ask a question about the document!", placeholder="Can you give me a short summary?", disabled=not uploaded_file)

    if uploaded_file and question:
        # Process the uploaded file and the question
        document = uploaded_file.read().decode()
        messages = [{"role": "user", "content": f"Here's a document: {document} \n\n---\n\n {question}"}]

        # Generate an answer using the OpenAI API (stream=True for streaming responses)
        stream = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            stream=True
        )

        # Stream the response to the app
        st.write_stream(stream)
else:
    st.info("Please add your OpenAI API key to continue.", icon="🗝️")


# Exploratory Data Analysis (EDA) Section
st.subheader("Exploratory Data Analysis (EDA)")

# Load the dataset via file uploader
uploaded_data_file = st.file_uploader("Upload a dataset (.csv)", type=["csv"])

if uploaded_data_file:
    data = pd.read_csv(uploaded_data_file)

    # Show the raw data if the checkbox is selected
    if st.checkbox('Show Raw Data'):
        st.write(data)

    # Show basic dataset information
    st.subheader('Basic Information')
    st.write('Number of rows and columns:', data.shape)
    st.write('Data types:')
    st.write(data.dtypes)

    # Show summary statistics
    st.subheader('Summary Statistics')
    st.write(data.describe())

    # Select a feature for plotting
    st.subheader('Plot Feature Distributions')
    column_to_plot = st.selectbox('Select a feature to plot', data.columns)

    # Plot a histogram for the selected column
    if column_to_plot:
        plt.figure(figsize=(8, 6))
        sns.histplot(data[column_to_plot].dropna(), kde=True)
        plt.title(f'Distribution of {column_to_plot}')
        st.pyplot(plt)

    # Filtering by year (if there is a "Year" column)
    if 'Year' in data.columns:
        st.subheader('Filter Data by Year')
        years = st.multiselect('Select years to filter by', data['Year'].unique())
        if years:
            filtered_data = data[data['Year'].isin(years)]
            st.write(f'Filtered Data for the selected years: {years}')
            st.write(filtered_data)

    # Display basic summary for a selected column
    st.subheader('Summary of Selected Feature')
    feature = st.selectbox('Select a feature to summarize', data.columns)
    if feature:
        st.write(f'Summary statistics for {feature}:')
        st.write(data[feature].describe())
