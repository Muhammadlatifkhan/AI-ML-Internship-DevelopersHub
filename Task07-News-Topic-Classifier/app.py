import streamlit as st
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import os

# Define the category names mapping (AG News labels: 0=World, 1=Sports, 2=Business, 3=Sci/Tech)
LABELS = ["World", "Sports", "Business", "Sci/Tech"]

@st.cache_resource
def load_model():
    model_path = "./saved_model"
    if not os.path.exists(model_path):
        return None, None
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertForSequenceClassification.from_pretrained(model_path)
    model.eval()
    return tokenizer, model

st.set_page_config(page_title="News Topic Classifier", page_icon="📰")

st.title("📰 News Topic Classifier")
st.markdown("This app uses a fine-tuned BERT model to classify news headlines into four categories: **World, Sports, Business, Sci/Tech**.")

tokenizer, model = load_model()

if tokenizer is None or model is None:
    st.error("Model files not found! Please make sure you have placed the trained model files in a folder named `saved_model` inside this directory. See README.md for instructions.")
else:
    text_input = st.text_area("Enter a news headline or short summary:", height=100)
    
    if st.button("Classify"):
        if text_input.strip() == "":
            st.warning("Please enter some text to classify.")
        else:
            with st.spinner("Classifying..."):
                inputs = tokenizer(text_input, return_tensors="pt", truncation=True, padding=True, max_length=128)
                with torch.no_grad():
                    outputs = model(**inputs)
                logits = outputs.logits
                predicted_class_id = logits.argmax().item()
                prediction = LABELS[predicted_class_id]
                
                # Calculate probabilities
                probs = torch.nn.functional.softmax(logits, dim=-1)[0]
                confidence = probs[predicted_class_id].item() * 100
                
            st.success(f"**Predicted Topic:** {prediction}")
            st.info(f"**Confidence:** {confidence:.2f}%")
