
import streamlit as st
import pickle
import re
import string
import random


# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="AI-Powered Chatbot",
    page_icon="🤖",
    layout="centered"
)


# ==========================================
# LOAD MODEL FILES
# ==========================================

with open("chatbot_model.pkl", "rb") as file:
    model = pickle.load(file)

with open("tfidf_vectorizer.pkl", "rb") as file:
    vectorizer = pickle.load(file)

with open("label_encoder.pkl", "rb") as file:
    label_encoder = pickle.load(file)

with open("response_mapping.pkl", "rb") as file:
    response_mapping = pickle.load(file)


# ==========================================
# TEXT CLEANING
# ==========================================

def clean_text(text):

    text = str(text).lower()

    text = re.sub(
        r"http\S+|www\S+|https\S+",
        "",
        text
    )

    text = re.sub(
        r"\S+@\S+",
        "",
        text
    )

    text = re.sub(
        r"\d+",
        "",
        text
    )

    text = text.translate(
        str.maketrans(
            "",
            "",
            string.punctuation
        )
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


# ==========================================
# CHATBOT
# ==========================================

def chatbot(user_message):

    cleaned = clean_text(user_message)

    vector = vectorizer.transform(
        [cleaned]
    )

    predicted_label = model.predict(
        vector
    )[0]

    predicted_intent = label_encoder.inverse_transform(
        [predicted_label]
    )[0]

    responses = response_mapping.get(
        predicted_intent,
        []
    )

    if responses:
        response = random.choice(responses)
    else:
        response = (
            "Sorry, I don't have an answer "
            "for that question."
        )

    return predicted_intent, response


# ==========================================
# CUSTOM CSS
# ==========================================

st.markdown("""
<style>

.title {
    text-align: center;
    font-size: 38px;
    font-weight: bold;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    color: gray;
    margin-bottom: 25px;
}

</style>
""", unsafe_allow_html=True)


# ==========================================
# HEADER
# ==========================================

st.markdown(
    '<div class="title">🤖 AI-Powered Chatbot</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">College Support Assistant</div>',
    unsafe_allow_html=True
)


# ==========================================
# SESSION STATE
# ==========================================

if "messages" not in st.session_state:
    st.session_state.messages = []


# ==========================================
# CHAT HISTORY
# ==========================================

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.write(
            message["content"]
        )


# ==========================================
# USER INPUT
# ==========================================

user_input = st.chat_input(
    "Ask your question..."
)


if user_input:

    # User message

    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    with st.chat_message("user"):
        st.write(user_input)


    # Bot response

    intent, response = chatbot(
        user_input
    )

    st.session_state.messages.append({
        "role": "assistant",
        "content": response
    })

    with st.chat_message("assistant"):
        st.write(response)


# ==========================================
# SIDEBAR
# ==========================================

with st.sidebar:

    st.header("🤖 Chatbot")

    st.write(
        "AI-powered college support assistant."
    )

    st.divider()

    st.subheader("Features")

    st.write("✓ NLP")
    st.write("✓ TF-IDF")
    st.write("✓ Intent Classification")
    st.write("✓ Machine Learning")
    st.write("✓ Automated Responses")

    st.divider()

    if st.button("Clear Chat"):

        st.session_state.messages = []

        st.rerun()
