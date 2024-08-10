import streamlit as st
import pandas as pd
import ast
import numpy as np
from openai.embeddings_utils import cosine_similarity
import openai

openai.api_key =  st.secrets["mykey"]

# Load the dataset
df = pd.read_csv("qa_dataset_with_embeddings.csv")

# Convert the string embeddings back to lists
df['Question_Embedding'] = df['Question_Embedding'].apply(ast.literal_eval)

# Function to get the embedding of a user question
def get_embedding(text, model="text-embedding-ada-002"):
    return openai.Embedding.create(input=[text], model=model)['data'][0]['embedding']

# Function to find the best answer
def find_best_answer(user_question):
    # Get embedding for the user's question
    user_question_embedding = get_embedding(user_question)

    # Calculate cosine similarities for all questions in the dataset
    df['Similarity'] = df['Question_Embedding'].apply(lambda x: cosine_similarity(x, user_question_embedding))

    # Find the most similar question and get its corresponding answer
    most_similar_index = df['Similarity'].idxmax()
    max_similarity = df['Similarity'].max()

    # Set a similarity threshold to determine if a question is relevant enough
    similarity_threshold = 0.6  # Adjust this value as needed

    if max_similarity >= similarity_threshold:
        best_answer = df.loc[most_similar_index, 'Answer']
        return best_answer, max_similarity
    else:
        return "I apologize, but I don't have information on that topic yet. Could you please ask other questions?", max_similarity

# Streamlit interface
st.title("Health FAQ Assistant")

# Text input for user's question
user_question = st.text_input("Enter your question:")

# Button to search for an answer
if st.button('Find Answer', key='find_answer'):
    if user_question:
        answer, similarity_score = find_best_answer(user_question)
        st.write(f"Answer: {answer}")
        st.write(f"Similarity Score: {similarity_score:.2f}")
    else:
        st.write("Please enter a question.")

# Clear button
if st.button('Clear', key='clear'):
    user_question = ""
    st.experimental_rerun()  # This will reset the input field

# Rating the answer's helpfulness
if user_question:
    st.write("Was this answer helpful?")
    rating = st.radio("", ('Yes', 'No'), key='rating')

    if rating == 'Yes':
        st.write("Thank you for your feedback!")
    elif rating == 'No':
        st.write("Sorry about that. We’ll work on improving our answers.")

# Common FAQs section in the sidebar
st.sidebar.title("Common FAQs")

# Search bar to filter questions
faq_search = st.sidebar.text_input("Search FAQs:", key='faq_search')

# Filter FAQs based on the search input
filtered_faqs = df[df['Question'].str.contains(faq_search, case=False, na=False)]

# Display the filtered FAQs in the sidebar
st.sidebar.write("Top 10 Matching FAQs:")
for index, row in filtered_faqs.head(10).iterrows():
    st.sidebar.write(f"**Q:** {row['Question']}")
    st.sidebar.write(f"**A:** {row['Answer']}\n")
