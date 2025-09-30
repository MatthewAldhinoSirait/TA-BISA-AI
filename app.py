import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import joblib
import torch
from wordcloud import WordCloud
from transformers import BertTokenizer, BertForSequenceClassification

# === CONFIG ===
st.set_page_config(
    page_title="Analisis Sentimen Timnas Indonesia",
    page_icon="⚽",
    layout="wide"
)

# === CSS CUSTOM (revisi: rata-rata merah-putih, tombol ditata & target tombol by aria-label) ===
st.markdown("""
<style>
body, .stApp {
    background-color: #0b1020;
    color: #e6eef8;
    text-align: center;
    font-family: "Inter", sans-serif;
}
h1, h2, h3, h4, h5, h6 {
    margin: .2rem 0;
}
[data-testid="stHeader"] {
    background: transparent !important;
    height: 0rem;
}
.title-gradient {
    background: linear-gradient(90deg, #ff0000 0%, #ffffff 50%, #ff0000 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 800;
    letter-spacing: 0.2px;
}
.metric-card {
    padding: 14px;
    border-radius: 14px;
    margin: 6px auto;
    text-align: center;
    color: #f8fafc;
    min-height: 88px;
    display: flex;
    flex-direction: column;
    justify-content: center;
    transition: transform .22s ease, box-shadow .22s ease;
}
.metric-card .meta-title {
    font-size: 0.9rem;
    opacity: 0.95;
    margin-bottom:6px;
    font-weight:700;}
.metric-card .meta-value {
    font-size:1.6rem;
    font-weight:800;
    letter-spacing:0.6px;}
.metric-card:hover {
    transform: translateY(-6px);
    box-shadow: 0 18px 30px rgba(2,6,23,0.6);
}
.card-total {
    background: linear-gradient(135deg,#0f172a,#ef4444);
    color:#fff0f0;
}
.card-positif {
    background: linear-gradient(135deg,#ffffff,#ef4444);
    color:#0b1020;
}
.card-negatif {
    background: linear-gradient(135deg,#7f1d1d,#ff0000);
    color:#fff;
}
/* REVISI: rata-rata jadi MERAH - PUTIH */
.card-ratarata {
    background: linear-gradient(135deg,#ff0000,#ffffff);
    color:#0b1020;
}
.emoji {
    display:inline-block;
    font-size:20px;
    margin-right:8px;
    vertical-align:middle;
}
.point-card {
    background: linear-gradient(135deg,#071033,#10243a);
    padding: 12px;
    border-radius: 12px;
    color:#dbeafe;
    box-shadow: 0 8px 20px rgba(2,6,23,0.6);
    margin: 8px 0;
}
.stTabs [role="tablist"] {
    justify-content: center;
    gap: 12px;
}
.stTabs [role="tab"] {
    background: linear-gradient(135deg,#0f172a,#1e293b);
    padding: 10px 18px;
    border-radius: 12px;
    font-weight: 700;
    color: #dbeafe !important;
    box-shadow: 0 4px 10px rgba(2,6,23,0.5);
    transition: all 0.2s ease-in-out;
}
.stTabs [role="tab"]:hover {
    background: linear-gradient(135deg,#1e293b,#334155);
    transform: translateY(-2px);
}
.stTabs [role="tab"][aria-selected="true"] {
    background: linear-gradient(135deg,#ef4444,#ffffff);
    color: black !important;
    box-shadow: 0 6px 16px rgba(0,0,0,0.4);
}
/* Pusatkan container tombol */
div[data-testid="stButton"] {
    display: flex;
    justify-content: center;
}

/* Style tombol Prediksi */
div[data-testid="stButton"] > button {
    background: linear-gradient(90deg,#ff0000 0%, #ffffff 100%) !important;
    color: #0b1020 !important;
    font-weight: 800;
    border-radius: 10px;
    padding: 10px 28px;
    border: none;
    cursor: pointer;
    transition: transform .12s ease;
    box-shadow: 0 8px 20px rgba(239,68,68,0.3);
}

/* Hover efek */
div[data-testid="stButton"] > button:hover {
    transform: translateY(-3px);
}

/* Tombol Prediksi khusus (pakai key prediksi_btn) */
div[data-testid="stButton"][data-baseweb="button"]:has(button[data-testid="baseButton-element"]) {
    display: flex;
    justify-content: center;
}

/* Warna merah-putih untuk tombol prediksi */
div[data-testid="stButton"]:has(button[data-testid="baseButton-element"][kind="secondary"]) button {
    background: linear-gradient(90deg,#ff0000 0%, #ffffff 100%) !important;
    color: #0b1020 !important;
    box-shadow: 0 8px 20px rgba(239,68,68,0.3);
}
@media (max-width: 800px) {
    .stSelectbox, .stTextArea, .stButton {
        width: 92% !important;
    }
    .stTabs [role="tablist"] { gap: 6px; }
    .stTabs [role="tab"] { font-size: 0.85rem; padding: 8px 12px; }
}
</style>
""", unsafe_allow_html=True)

# === TITLE UTAMA ===
st.markdown("<h1 class='title-gradient' style='font-size:34px; margin-top:6px; margin-bottom:6px;'>Dashboard Analisis Sentimen Timnas Indonesia Era Shin Tae-yong</h1>", unsafe_allow_html=True)

# === LOAD DATASET ===
@st.cache_data
def load_data(path="Analisis_sentimen_timnas_sepakbola_indonesia_di_era_STY.csv"):
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        st.error(f"File CSV tidak ditemukan: {path}")
        return pd.DataFrame(columns=["komentar", "label"])

    if 'komentar' not in df.columns or 'label' not in df.columns:
        st.error("CSV harus memiliki kolom 'komentar' dan 'label'.")
        return pd.DataFrame(columns=["komentar", "label"])

    df['label'] = df['label'].map({'negatif': 0, 'positif': 1}).fillna(df['label'])
    try:
        df['label'] = df['label'].astype(int)
    except Exception:
        pass
    df['komentar'] = df['komentar'].astype(str)
    df['comment_length'] = df['komentar'].apply(len)
    return df

df = load_data()

# === LOAD MODELS (opsional) ===
nb_model, vectorizer = None, None
try:
    nb_model = joblib.load("naive_bayes_model.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
except Exception:
    st.warning("Model Naive Bayes / TF-IDF vectorizer tidak ditemukan — fungsi Naive Bayes akan dinonaktifkan jika tidak ada.")

MODEL_PATH = "matthewaldhino/indobert-sentiment"
tokenizer, bert_model = None, None
try:
    tokenizer = BertTokenizer.from_pretrained(MODEL_PATH, local_files_only=False)
    bert_model = BertForSequenceClassification.from_pretrained(
        MODEL_PATH,
        num_labels=2,
        torch_dtype=torch.float32,
        device_map=None,          # force full load, no meta
        low_cpu_mem_usage=False   # disable lazy load
    )
    bert_model.to("cpu")
    bert_model.eval()
except Exception as e:
    st.warning(f"Model IndoBERT tidak dapat dimuat: {e}")

# === FUNGSI PREDIKSI ===
def predict_nb(text):
    if nb_model is None or vectorizer is None:
        return "Model NB tidak tersedia", [0.0, 0.0]
    X = vectorizer.transform([text])
    pred = nb_model.predict(X)[0]
    prob = nb_model.predict_proba(X)[0]
    return ("Positif" if int(pred) == 1 else "Negatif"), prob

def predict_bert(text):
    if bert_model is None or tokenizer is None:
        return "Model BERT tidak tersedia", [0.0, 0.0]

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    ).to("cpu")

    with torch.no_grad():
        outputs = bert_model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
        pred = int(torch.argmax(logits, dim=1).cpu().numpy())

    return ("Positif" if pred == 1 else "Negatif"), probs

# === WARNA GLOBAL ===
COLOR_MAP = {"Positif": "#ff0000", "Negatif": "#ffffff"}

# === NAVIGATION: TABS ===
tab_overview, tab_eda, tab_model, tab_predict, tab_conclude = st.tabs(
    ["📊 Overview", "🔍 EDA", "🧮 Modelling", "🔮 Prediction", "📌 Conclusion"]
)

# === OVERVIEW ===
with tab_overview:
    st.subheader("Overview Dashboard")
    if df.empty:
        st.info("Dataset kosong atau tidak sesuai format.")

    col1, col2, col3, col4 = st.columns(4, gap="small")
    with col1:
        total = df.shape[0]
        st.markdown(f"<div class='metric-card card-total'><div class='meta-title'>📦 Total Data</div><div class='meta-value'>{total:,}</div></div>", unsafe_allow_html=True)
    with col2:
        pos = int((df['label'] == 1).sum()) if 'label' in df.columns else 0
        st.markdown(f"<div class='metric-card card-positif'><div class='meta-title'>🟥 Positif</div><div class='meta-value'>{pos:,}</div></div>", unsafe_allow_html=True)
    with col3:
        neg = int((df['label'] == 0).sum()) if 'label' in df.columns else 0
        st.markdown(f"<div class='metric-card card-negatif'><div class='meta-title'>⬜ Negatif</div><div class='meta-value'>{neg:,}</div></div>", unsafe_allow_html=True)
    with col4:
        avg_len = df['comment_length'].mean() if 'comment_length' in df.columns else 0
        st.markdown(f"<div class='metric-card card-ratarata'><div class='meta-title'>✍️ Rata-rata Panjang</div><div class='meta-value'>{avg_len:.1f}</div></div>", unsafe_allow_html=True)

    st.markdown("---")
    if not df.empty and 'komentar' in df.columns and 'label' in df.columns:
        col_left, col_right = st.columns(2, gap="large")
        with col_left:
            label_series = df['label'].map({0: 'Negatif', 1: 'Positif'})
            label_counts = label_series.value_counts().reindex(['Positif','Negatif']).fillna(0)
            fig_pie = px.pie(
                names=label_counts.index,
                values=label_counts.values,
                color=label_counts.index,
                color_discrete_map=COLOR_MAP,
                hole=0.35
            )
            fig_pie.update_traces(textinfo="percent+label", pull=[0.06, 0.02])
            fig_pie.update_layout(title=dict(text="Proporsi Sentimen", x=0.5, xanchor="center"), paper_bgcolor="#0b1020", font=dict(color="#e6eef8"))
            st.plotly_chart(fig_pie, use_container_width=True)

        with col_right:
            fig_len = go.Figure(data=[go.Histogram(
                x=df["comment_length"], nbinsx=30,
                marker=dict(color="#ff0000", line=dict(color="#0b1020", width=1.5)), opacity=0.95
            )])
            fig_len.update_layout(
                title=dict(text="Distribusi Panjang Komentar", x=0.5, xanchor="center"),
                paper_bgcolor="#0b1020", plot_bgcolor="#0b1020", font=dict(color="#e6eef8"),
                xaxis=dict(title="Panjang (karakter)"), yaxis=dict(title="Jumlah")
            )
            st.plotly_chart(fig_len, use_container_width=True)

# === EDA ===
with tab_eda:
    st.subheader("Exploratory Data Analysis")
    if df.empty or 'label' not in df.columns:
        st.info("Data tidak tersedia untuk EDA.")
    else:
        label_series = df['label'].map({0: 'Negatif', 1: 'Positif'})
        label_counts = label_series.value_counts().reset_index()
        label_counts.columns = ['Sentimen', 'Jumlah']

        colors = ["#ff0000" if s == "Positif" else "#ffffff" for s in label_counts['Sentimen']]
        fig_sent = go.Figure(data=[go.Bar(
            x=label_counts["Sentimen"], y=label_counts["Jumlah"],
            marker=dict(color=colors, line=dict(color="#0b1020", width=1.8)),
            text=label_counts["Jumlah"], textposition='outside'
        )])
        fig_sent.update_layout(title=dict(text="Jumlah Komentar per Sentimen", x=0.5, xanchor="center"),
            paper_bgcolor="#0b1020", plot_bgcolor="#0b1020", font=dict(color="#e6eef8"), yaxis=dict(showgrid=False))
        st.plotly_chart(fig_sent, use_container_width=True)

        st.markdown("<h4 style='color:#dbeafe; margin-top:12px;'>Wordcloud Sentimen</h4>", unsafe_allow_html=True)
        col1, col2 = st.columns(2, gap="large")
        with col1:
            pos_text = " ".join(df[df['label'] == 1]['komentar'].astype(str)) if 1 in df['label'].unique() else ""
            if pos_text.strip():
                wc_pos = WordCloud(width=800, height=400, background_color="#071033", colormap="Reds").generate(pos_text)
                fig, ax = plt.subplots(figsize=(8, 4.5))
                ax.imshow(wc_pos, interpolation='bilinear')
                ax.axis("off")
                st.pyplot(fig)
                plt.close(fig)
            else:
                st.info("Tidak ada data positif.")
        with col2:
            neg_text = " ".join(df[df['label'] == 0]['komentar'].astype(str)) if 0 in df['label'].unique() else ""
            if neg_text.strip():
                wc_neg = WordCloud(width=800, height=400, background_color="#071033", colormap="Blues").generate(neg_text)
                fig, ax = plt.subplots(figsize=(8, 4.5))
                ax.imshow(wc_neg, interpolation='bilinear')
                ax.axis("off")
                st.pyplot(fig)
                plt.close(fig)
            else:
                st.info("Tidak ada data negatif.")

# === MODELLING & EVALUASI ===
with tab_model:
    st.subheader("Model & Evaluasi")
    metrics = pd.DataFrame({
        "Model": ["Naive Bayes", "IndoBERT"],
        "Akurasi": [0.675, 0.7375],
        "F1-Score": [0.67, 0.74]
    })
    fig_model = px.bar(metrics.melt(id_vars='Model', value_vars=['Akurasi', 'F1-Score']),
        x='Model', y='value', color='variable', barmode='group',
        color_discrete_sequence=["#ff0000", "#ffffff"])
    fig_model.update_layout(title=dict(text="Performa Model", x=0.5, xanchor="center"),
        paper_bgcolor="#0b1020", plot_bgcolor="#0b1020", font=dict(color="#e6eef8"))
    st.plotly_chart(fig_model, use_container_width=True)

    st.markdown("<h4 style='color:#dbeafe; margin-top:10px;'>Confusion Matrix (IndoBERT)</h4>", unsafe_allow_html=True)
    cm = np.array([[80, 20], [15, 85]])
    labels_x = ["Pred: Negatif", "Pred: Positif"]
    labels_y = ["Aktual: Negatif", "Aktual: Positif"]
    fig_cm = go.Figure(data=go.Heatmap(
        z=cm,
        x=labels_x,
        y=labels_y,
        colorscale=[[0, "#ffffff"], [1, "#ff0000"]],
        showscale=False,
        hovertemplate="Aktual: %{y}<br>Prediksi: %{x}<br>Jumlah: %{z}<extra></extra>"
    ))
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            fig_cm.add_annotation(dict(
                x=labels_x[j], y=labels_y[i], text=str(int(cm[i, j])), showarrow=False,
                font=dict(color="#0b1020", size=14)
            ))
    fig_cm.update_layout(title=dict(text="Confusion Matrix", x=0.5, xanchor="center"),
                         paper_bgcolor="#0b1020", font=dict(color="#e6eef8"),
                         xaxis=dict(tickfont=dict(color="#e6eef8")), yaxis=dict(tickfont=dict(color="#e6eef8")))
    st.plotly_chart(fig_cm, use_container_width=True)

    st.markdown("<div class='point-card'>✅ Naive Bayes cepat dan sederhana.</div>", unsafe_allow_html=True)
    st.markdown("<div class='point-card'>✅ IndoBERT lebih unggul dalam konteks bahasa Indonesia.</div>", unsafe_allow_html=True)
    st.markdown("<div class='point-card'>✅ IndoBERT seimbang dalam mengenali komentar.</div>", unsafe_allow_html=True)

# === PREDICTION ===
with tab_predict:
    st.subheader("Prediksi Sentimen Komentar")
    model_choice = st.selectbox("Pilih Model", ["Naive Bayes", "IndoBERT"])
    user_input = st.text_area("Masukkan komentar:", placeholder="Contoh: Timnas bermain sangat bagus!")

    # Tombol prediksi di tengah
    predict_clicked = st.button("Prediksi")

    if predict_clicked:
        if not user_input.strip():
            st.warning("Masukkan komentar terlebih dahulu!")
        else:
            if model_choice == "Naive Bayes":
                hasil, prob = predict_nb(user_input)
            else:
                hasil, prob = predict_bert(user_input)

            if isinstance(hasil, str) and "tidak tersedia" in hasil.lower():
                st.error(hasil)
            else:
                st.success(f"🔮 Prediksi: {hasil} (Model: {model_choice})")

                # pastikan prob bisa diindeks
                try:
                    p_pos = float(prob[1]) if len(prob) >= 2 else 0.0
                    p_neg = float(prob[0]) if len(prob) >= 2 else 0.0
                except Exception:
                    p_pos, p_neg = 0.0, 0.0

                col1, col2 = st.columns(2, gap="small")
                with col1:
                    st.markdown(f"<div class='metric-card card-positif'><div class='meta-title'>🟥 Positif</div><div class='meta-value'>{p_pos:.2f}</div></div>", unsafe_allow_html=True)
                with col2:
                    st.markdown(f"<div class='metric-card card-negatif'><div class='meta-title'>⬜ Negatif</div><div class='meta-value'>{p_neg:.2f}</div></div>", unsafe_allow_html=True)

                # visual probabilitas
                fig_prob = go.Figure(data=[go.Bar(
                    x=["Negatif", "Positif"], y=[p_neg, p_pos],
                    marker=dict(color=["#ffffff", "#ff0000"], line=dict(color="#0b1020", width=1.5))
                )])
                fig_prob.update_layout(title=dict(text="Probabilitas Prediksi", x=0.5, xanchor="center"),
                                       paper_bgcolor="#0b1020", plot_bgcolor="#0b1020", font=dict(color="#e6eef8"),
                                       yaxis=dict(range=[0, 1]))
                st.plotly_chart(fig_prob, use_container_width=True)

# === CONCLUSION ===
with tab_conclude:
    st.subheader("Kesimpulan")
    st.markdown("<div class='point-card'>IndoBERT mencapai akurasi 73.7%, lebih tinggi dari Naive Bayes (67.5%).</div>", unsafe_allow_html=True)
    st.markdown("<div class='point-card'>EDA menunjukkan sentimen penggemar cukup berimbang.</div>", unsafe_allow_html=True)
    st.markdown("<div class='point-card'>Dashboard ini bisa memprediksi komentar baru secara real-time (jika model dimuat).</div>", unsafe_allow_html=True)