import streamlit as st
import pickle
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tensorflow.keras.models import load_model
from ngram_model import NGramModel  # Ensure this file exists

# -------------------------
# Load N-gram model
# -------------------------
with open("ngram_model.pkl", "rb") as f:
    ngram = pickle.load(f)

# -------------------------
# Load LSTM model & tokenizer
# -------------------------
lstm_model = load_model("lstm_model.h5")

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# -------------------------
# Load GPT-2
# -------------------------
gpt_tokenizer = AutoTokenizer.from_pretrained("distilgpt2", chat_template=None)
gpt_model = AutoModelForCausalLM.from_pretrained("distilgpt2")
gpt_model.eval()

# -------------------------
# Prediction functions
# -------------------------
def predict_ngram(context_words, top_k=3):
    context_ids = [ngram.word_index.get(w.lower(), ngram.word_index.get("<OOV>", 0))
                   for w in context_words[-(ngram.n-1):]]
    top_ids, top_probs = ngram.predict_next(context_ids, top_k=top_k)
    top_words = [tokenizer.index_word.get(i, "<OOV>") for i in top_ids]
    return list(zip(top_words, top_probs))

def predict_lstm_func(text, seq_length=15, top_k=3):
    tokens = tokenizer.texts_to_sequences([text])[0]
    tokens = tokens[-seq_length:]
    if len(tokens) < seq_length:
        tokens = [0]*(seq_length - len(tokens)) + tokens

    preds = lstm_model.predict(np.array([tokens]), verbose=0)[0]
    top_ids = preds.argsort()[-top_k:][::-1]
    top_probs = preds[top_ids]
    top_words = [tokenizer.index_word.get(i, "<OOV>") for i in top_ids]
    return list(zip(top_words, top_probs))

def predict_gpt2_func(text, top_k=3):
    inputs = gpt_tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        logits = gpt_model(**inputs).logits[0, -1]
    probs = torch.softmax(logits, dim=-1)
    topk = torch.topk(probs, k=top_k)
    words = [gpt_tokenizer.decode([i]).strip() for i in topk.indices]
    return list(zip(words, topk.values.tolist()))

# -------------------------
# Streamlit UI
# -------------------------
st.title("Next-Word Predictor: N-gram, LSTM, GPT-2")

user_text = st.text_input("Enter your text:")
top_k = st.slider("Top-K predictions", 1, 5, 3)
model_choice = st.selectbox("Choose model:", ["N-gram", "LSTM", "GPT-2"])

if st.button("Predict"):
    if not user_text.strip():
        st.warning("Enter some text first.")
    else:
        if model_choice == "N-gram":
            predictions = predict_ngram(user_text.split(), top_k=top_k)
        elif model_choice == "LSTM":
            predictions = predict_lstm_func(user_text, top_k=top_k)
        elif model_choice == "GPT-2":
            predictions = predict_gpt2_func(user_text, top_k=top_k)
        else:
            predictions = []

        st.write("### Top Predictions:")
        for i, (word, prob) in enumerate(predictions, 1):
            st.write(f"{i}. {word} — {prob:.4f}")