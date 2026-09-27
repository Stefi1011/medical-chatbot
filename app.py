import streamlit as st
import pandas as pd
import numpy as np
import torch

from sentence_transformers import SentenceTransformer, util


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Medical Q&A Chatbot",
    page_icon="🩺",
    layout="centered"
)


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():
    return SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )


model = load_model()


# =========================================================
# LOAD KNOWLEDGE BASE
# =========================================================

@st.cache_data
def load_data():
    return pd.read_csv("knowledge_base.csv")


knowledge_base = load_data()


# =========================================================
# LOAD EMBEDDINGS
# =========================================================

@st.cache_resource
def load_embeddings():
    embeddings = np.load(
        "question_embeddings.npy"
    )

    return torch.tensor(
        embeddings,
        dtype=torch.float32
    )


question_embeddings = load_embeddings()


# =========================================================
# SEARCH FUNCTION
# =========================================================

def search_top_k(user_question, k=3):

    query_embedding = model.encode(
        user_question,
        convert_to_tensor=True
    )

    similarities = util.cos_sim(
        query_embedding,
        question_embeddings
    )[0]

    top_results = similarities.topk(k)

    results = []

    for score, idx in zip(
        top_results.values,
        top_results.indices
    ):

        idx = idx.item()

        results.append({
            "qtype":
                knowledge_base.iloc[idx]["qtype"],

            "question":
                knowledge_base.iloc[idx]["Question"],

            "answer":
                knowledge_base.iloc[idx]["Answer"],

            "similarity":
                score.item()
        })

    return results


# =========================================================
# CHATBOT
# =========================================================

CONFIDENCE_THRESHOLD = 0.50


def medical_chatbot(user_question):

    results = search_top_k(
        user_question,
        k=3
    )

    best_result = results[0]

    if (
        best_result["similarity"]
        < CONFIDENCE_THRESHOLD
    ):

        return {
            "status": "low_confidence",

            "answer":
                "Sorry, I could not find sufficiently "
                "relevant medical information for "
                "your question.",

            "matched_question":
                best_result["question"],

            "qtype":
                best_result["qtype"],

            "similarity":
                best_result["similarity"],

            "related_questions": []
        }


    return {
        "status": "success",

        "answer":
            best_result["answer"],

        "matched_question":
            best_result["question"],

        "qtype":
            best_result["qtype"],

        "similarity":
            best_result["similarity"],

        "related_questions": [
            result["question"]
            for result in results[1:]
        ]
    }


# =========================================================
# STREAMLIT INTERFACE
# =========================================================

st.title("🩺 Medical Q&A Chatbot")

st.write(
    "Ask a medical question and the chatbot "
    "will retrieve relevant information "
    "from the MedQuAD dataset."
)

st.info(
    "This chatbot is for educational and "
    "informational purposes only and is not "
    "a substitute for professional medical advice."
)


# =========================================================
# CHAT HISTORY
# =========================================================

if "messages" not in st.session_state:
    st.session_state.messages = []


for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# =========================================================
# USER INPUT
# =========================================================

user_question = st.chat_input(
    "Ask a medical question..."
)


if user_question:

    # Simpan pesan user
    st.session_state.messages.append({
        "role": "user",
        "content": user_question
    })

    with st.chat_message("user"):
        st.markdown(user_question)


    # Jalankan chatbot
    result = medical_chatbot(
        user_question
    )


    # Tampilkan jawaban
    with st.chat_message("assistant"):

        st.markdown(
            result["answer"]
        )


        if result["status"] == "success":

            with st.expander(
                "Retrieval Details"
            ):

                st.write(
                    "**Matched Question:**",
                    result["matched_question"]
                )

                st.write(
                    "**Category:**",
                    result["qtype"]
                )

                st.write(
                    "**Similarity:**",
                    f'{result["similarity"]:.4f}'
                )


                if result["related_questions"]:

                    st.write(
                        "**Related Questions:**"
                    )

                    for question in (
                        result[
                            "related_questions"
                        ]
                    ):

                        st.write(
                            "-",
                            question
                        )


    # Simpan jawaban chatbot
    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"]
    })
