import openai
import streamlit as st

from dotenv import load_dotenv
import os

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def get_response(prompt, model="gpt-3.5-turbo"):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is machine learning?"}
    ]
)
    except Exception as e:
        return f"Error: {e}"

# Streamlit UI
st.title("Interactive Prompt Engineering with LLM")

st.sidebar.header("Prompt Templates")
templates = {
    "Creative Writing": "Write a story about a [subject].",
    "Summarization": "Summarize the following text: [text].",
    "Question Answering": "Answer this question: [question].",
    "Coding Help": "Write Python code to [task].",
}
template_choice = st.sidebar.selectbox(
    "Choose a template",
    options=["Custom"] + list(templates.keys())
)

# If a template is selected, prefill the prompt box
if template_choice != "Custom":
    prompt = templates[template_choice]
else:
    prompt = ""

# Prompt input box
user_prompt = st.text_area("Enter your prompt:", value=prompt, height=150)

# Button to get LLM response
if st.button("Get Response"):
    with st.spinner("Generating response..."):
        response = get_response(user_prompt)
        st.success("Response generated!")
        st.write(response)

# Comparison mode
st.header("Comparison Mode (Optional)")
st.write("Test multiple prompts side-by-side.")
col1, col2 = st.columns(2)

with col1:
    prompt1 = st.text_area("Prompt 1", height=100, key="prompt1")
    if st.button("Get Response for Prompt 1", key="button1"):
        response1 = get_response(prompt1)
        st.write(response1)

with col2:
    prompt2 = st.text_area("Prompt 2", height=100, key="prompt2")
    if st.button("Get Response for Prompt 2", key="button2"):
        response2 = get_response(prompt2)
        st.write(response2)

# Footer
st.sidebar.write("Developed for exploring prompt engineering with LLMs.")
