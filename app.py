import os
import re
import string
import gdown
import streamlit as st
import torch
import pandas as pd
from transformers import BertTokenizer, BertConfig, BertForSequenceClassification

# ===================== MODEL URL =====================
# Model Aspek
ASPEK_FOLDER = "transportasi_model"
ASPEK_MODEL_FILES = {
    "config.json": "https://drive.google.com/uc?id=1YxNsSODaGHkcur_j0kBbhLoMMoRDegiO",
    "best_model.pt": "https://drive.google.com/uc?id=1AXwYWwJebPaMkEohZcC6DxRMXQ50Mx2F",
    "vocab.txt": "https://drive.google.com/uc?id=132x3vGdctS7EYXkhSswKsxJXmYp7ct-F",
    "special_tokens_map.json": "https://drive.google.com/uc?id=1g3Zyx5c810WXb1ExN8OSOmM89drtO6rd",
    "tokenizer_config.json": "https://drive.google.com/uc?id=1ryaJ5ZuQelZtJpFZ6qIFaheFnnqTKkQ4"
}

# Model Sentimen
SENTIMEN_FOLDER = "sentimen_transportasi_model"
SENTIMEN_MODEL_FILES = {
    "config.json": "https://drive.google.com/uc?id=1iROZ-TiYdurNwpsF7nJYNaTPYLYjm2Ac",
    "best_model.pt": "https://drive.google.com/uc?id=1vuz7WAe6cz32E83wikcxxbDcOHAA-4tr",
    "vocab.txt": "https://drive.google.com/uc?id=1cAK50cqZpleMluh1HR8DjBCO9mTgqfVr",
    "special_tokens_map.json": "https://drive.google.com/uc?id=1J05-fPEUd__FxN0pImJ9Aw4QWvNv_C1-",
    "tokenizer_config.json": "https://drive.google.com/uc?id=1V9brPqA9PwHZRnf25_l39OZMVGj8cTG1"
}

KAMUS_CSV_URL = "https://drive.google.com/uc?id=1fGWZu5qVYJa-pv078spaLE4urs5zDDPV"
KAMUS_PATH = "kamus.csv"

# ===================== FUNGSI BANTUAN =====================
def download_model(model_folder, model_files):
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
    for filename, url in model_files.items():
        path = os.path.join(model_folder, "pytorch_model.bin" if filename == "best_model.pt" else filename)
        if not os.path.exists(path):
            with st.spinner(f"Mengunduh {filename}..."):
                gdown.download(url, path, quiet=False)

def download_kamus():
    if not os.path.exists(KAMUS_PATH):
        with st.spinner("Mengunduh kamus slang..."):
            gdown.download(KAMUS_CSV_URL, KAMUS_PATH, quiet=False)

@st.cache_resource(show_spinner=True)
def load_tokenizer(folder):
    return BertTokenizer.from_pretrained(folder)

@st.cache_resource(show_spinner=True)
def load_model(folder):
    config = BertConfig.from_pretrained(folder)
    model = BertForSequenceClassification.from_pretrained(folder, config=config)
    model.eval()
    return model

@st.cache_resource
def load_kamus():
    df = pd.read_csv(KAMUS_PATH)
    return dict(zip(df['slang'], df['formal']))

def preprocess(text, kamus_slang):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|@\S+|#\S+', '', text)
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = ' '.join([kamus_slang.get(word, word) for word in text.split()])
    return text.strip()

# ===================== APLIKASI STREAMLIT =====================
def main():
    st.set_page_config(page_title="Prediksi Aspek dan Sentimen", layout="wide")
    st.title("🕋 Prediksi Aspek & Sentimen - Layanan Transportasi Haji")

    download_model(ASPEK_FOLDER, ASPEK_MODEL_FILES)
    download_model(SENTIMEN_FOLDER, SENTIMEN_MODEL_FILES)
    download_kamus()

    kamus_slang = load_kamus()

    tokenizer_aspek = load_tokenizer(ASPEK_FOLDER)
    model_aspek = load_model(ASPEK_FOLDER)

    tokenizer_sentimen = load_tokenizer(SENTIMEN_FOLDER)
    model_sentimen = load_model(SENTIMEN_FOLDER)

    text = st.text_area("Masukkan teks ulasan atau komentar:", height=150)

    if st.button("Prediksi Aspek & Sentimen"):
        if not text.strip():
            st.warning("Masukkan teks dulu ya!")
            return

        cleaned = preprocess(text, kamus_slang)

        # Prediksi Aspek
        inputs_aspek = tokenizer_aspek(cleaned, return_tensors="pt", padding=True, truncation=True, max_length=128)
        with torch.no_grad():
            output_aspek = model_aspek(**inputs_aspek)
        pred_aspek = torch.argmax(output_aspek.logits, dim=1).item()

        if pred_aspek == 1:
            st.success("✅ Teks ini **termasuk aspek transportasi.**")

            # Prediksi Sentimen jika aspek sesuai
            inputs_sentimen = tokenizer_sentimen(cleaned, return_tensors="pt", padding=True, truncation=True, max_length=128)
            with torch.no_grad():
                output_sentimen = model_sentimen(**inputs_sentimen)
            pred_sentimen = torch.argmax(output_sentimen.logits, dim=1).item()

            if pred_sentimen == 2:
                st.success("Sentimen terhadap transportasi: **Positif**")
            elif pred_sentimen == 0:
                st.error("Sentimen terhadap transportasi: **Negatif**")
            else:
                st.warning("Sentimen tidak terdeteksi dengan pasti.")
        else:
            st.warning("⛔ Teks ini **tidak termasuk aspek transportasi.**")

if __name__ == "__main__":
    main()
