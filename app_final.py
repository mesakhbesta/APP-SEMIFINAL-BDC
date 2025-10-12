import streamlit as st
import pandas as pd
import re, time, io, tempfile, requests, numpy as np, soundfile as sf, whisper, emoji, joblib
from datetime import datetime, date
from pydub import AudioSegment
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.firefox.options import Options
from openai import OpenAI
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# OPTIONAL: aktifkan fallback scraping komentar via Apify
import streamlit as st
import os

# ==============================================================
# 🔒 LOAD TOKENS DARI SECRETS ATAU ENV (AMAN UNTUK STREAMLIT)
# ==============================================================

# Apify multi-token fallback
if "APIFY_TOKENS" in st.secrets:
    APIFY_TOKENS = [
        t.strip()
        for t in st.secrets["APIFY_TOKENS"].split(",")
        if t.strip()
    ]
else:
    APIFY_TOKENS = [
        t.strip()
        for t in os.getenv("APIFY_TOKENS", "").split(",")
        if t.strip()
    ]

# OpenAI key (Sumopod)
if "OPENAI_API_KEY" in st.secrets:
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
else:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# fallback kalau gak ada
if not APIFY_TOKENS:
    st.warning("⚠️ ")
if not OPENAI_API_KEY:
    st.warning("⚠️ ")

# Optional: aktifkan fallback scraping komentar via Apify
try:
    from apify_client import ApifyClient
except Exception:
    ApifyClient = None

# =========================
# CONFIG (set sekali)
# =========================
st.set_page_config(page_title="ReelTalk AI", page_icon="💬 ", layout="wide")
st.markdown("""
<style>
/* Scrollbar styling */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-thumb { background-color: rgba(255,255,255,0.2); border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background-color: rgba(255,255,255,0.4); }
</style>
""", unsafe_allow_html=True)

# ======================================================
# ⚙️ SETUP FIREFOX
# ======================================================
def create_driver():
    opts = Options()
    opts.add_argument("--headless")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    return webdriver.Firefox(options=opts)

# ======================================================
# 📊 SCRAPER METRIK INSTAGRAM (stabil wait + caption)
# ======================================================
def scrape_instagram_reel(url):
    # 🔹 regex baru yang benar-benar fleksibel
    match = re.search(r"(?:instagram\.com/)(?:[\w.-]+/)?reel/([A-Za-z0-9_-]+)", url)
    if not match:
        st.error(
            "❌ URL tidak valid. Pastikan mengandung '/reel/<ID>', misalnya:\n"
            "- https://www.instagram.com/reel/XXXXX/\n"
            "- https://www.instagram.com/<username>/reel/XXXXX/"
        )
        return None

    video_id = match.group(1)
    driver = create_driver()
    target_url = f"https://social-tracker.com/stats/instagram/reels/{video_id}"
    driver.get(target_url)
    wait = WebDriverWait(driver, 60)
    data = {"url": url, "reel_id": video_id}
    try:
        wait.until(EC.presence_of_element_located((By.TAG_NAME, "h3")))

        # Tunggu animasi angka stabil
        stable = False
        last_snapshot = []
        start_time = time.time()
        while not stable and (time.time() - start_time < 60):
            elems = driver.find_elements(By.CSS_SELECTOR, "p.mt-2.text-2xl.font-bold")
            snapshot = [e.text.strip() for e in elems if e.text.strip()]
            if snapshot == last_snapshot and len(snapshot) >= 4:
                stable = True
            else:
                last_snapshot = snapshot
                time.sleep(2)

        # Fullname & Username
        fullname, username = None, None
        try:
            h3_elems = driver.find_elements(By.CSS_SELECTOR, "h3.font-semibold.text-lg")
            for h in h3_elems:
                txt = h.text.strip()
                if txt and txt.lower() != "analytics":
                    fullname = txt
                    break
        except:
            pass
        data["fullname"] = fullname

        try:
            rel_elem = driver.find_element(By.CSS_SELECTOR, "div > h3.font-semibold.text-lg + p.text-gray-500.text-sm")
            txt_rel = rel_elem.text.strip()
            username = txt_rel if txt_rel.startswith("@") else None
        except:
            elems = driver.find_elements(By.CSS_SELECTOR, "p.text-gray-500.text-sm")
            for e in elems:
                t = e.text.strip()
                if t.startswith("@"):
                    username = t
                    break
        data["username"] = username

        # Caption
        try:
            cap_elem = driver.find_element(By.CSS_SELECTOR, "p.text-gray-800.dark\\:text-gray-200.leading-relaxed")
            data["caption"] = cap_elem.text.strip()
        except:
            data["caption"] = None

        # Duration
        try:
            dur_elem = driver.find_element(By.CSS_SELECTOR, "div.absolute.bottom-2.right-2.bg-black.bg-opacity-70.text-white.px-2.py-1.rounded.text-sm")
            data["duration"] = dur_elem.text.strip()
        except:
            data["duration"] = None

        # Main Stats
        try:
            elems_p = driver.find_elements(By.CSS_SELECTOR, "p.mt-2.text-2xl.font-bold")
            vals = [e.text.strip().replace(",", "").replace(".", "") for e in elems_p if e.text.strip()]
            if len(vals) >= 4:
                data["plays"] = int(vals[0]) if vals[0].isdigit() else None
                data["views"] = int(vals[1]) if vals[1].isdigit() else None
                data["likes"] = int(vals[2]) if vals[2].isdigit() else None
                data["comments"] = int(vals[3]) if vals[3].isdigit() else None
        except:
            data.update({"plays": None, "views": None, "likes": None, "comments": None})

        # Engagement
        try:
            elems_span = driver.find_elements(By.CSS_SELECTOR, "span.text-lg.font-bold")
            vals = [e.text.strip().replace("%", "").replace(",", ".") for e in elems_span if e.text.strip()]
            if len(vals) >= 1:
                engagement_raw = vals[0]
                try:
                    engagement_val = float(engagement_raw) / 100
                except:
                    engagement_val = None
                data["engagement"] = engagement_val
            else:
                data["engagement"] = None

            if data["engagement"] is None or (data["engagement"] is not None and (data["engagement"] > 1.0 or data["engagement"] < 0)):
                if data.get("plays"):
                    calc = ((data.get("likes") or 0) + (data.get("comments") or 0)) / data["plays"]
                    data["engagement"] = round(calc, 4)
        except:
            data["engagement"] = None

    except Exception as e:
        st.error(f"⚠️ Error scraping: {e}")

    driver.quit()
    return data

# ======================================================
# 🎧 AMBIL AUDIO & TRANSKRIPSI
# ======================================================
def get_audio_from_instagram(url):
    driver = create_driver()
    driver.get("https://reelsave.app/audio")
    try:
        input_box = WebDriverWait(driver, 60).until(EC.element_to_be_clickable((By.CSS_SELECTOR, "input[name='url']")))
        time.sleep(2)
        input_box.clear()
        input_box.send_keys(url)
        time.sleep(1)
        driver.find_element(By.CSS_SELECTOR, "button[type='submit']").click()

        WebDriverWait(driver, 180).until(EC.presence_of_element_located((By.XPATH, "//button[contains(., 'Download Audio')]")))
        audio_link_el = WebDriverWait(driver, 180).until(EC.presence_of_element_located((By.XPATH, "//a[contains(@href, 'videodropper')]")))
        audio_url = audio_link_el.get_attribute("href")
        driver.quit()
        return audio_url
    except Exception as e:
        driver.quit()
        st.error(f"❌ Gagal mengambil audio: {e}")
        return None

def download_file(url, suffix):
    tmp_path = tempfile.NamedTemporaryFile(delete=False, suffix=suffix).name
    r = requests.get(url)
    with open(tmp_path, "wb") as f:
        f.write(r.content)
    return tmp_path

def transcribe_audio(tmp_path):
    audio_data, sr = sf.read(tmp_path, dtype="float32")
    if len(audio_data.shape) > 1:
        audio_data = np.mean(audio_data, axis=1)
    if sr != 16000:
        import resampy
        audio_data = resampy.resample(audio_data, sr, 16000)
    model = whisper.load_model("base")
    result = model.transcribe(audio_data)
    return result["text"]

# ======================================================
# 💬 SENTIMEN ANALISIS KOMENTAR (MODEL LOAD)
# ======================================================
tfidf = joblib.load(r"tfidf_sentiment.joblib")
xgb = joblib.load(r"xgb_sentiment.joblib")
lbl = joblib.load(r"label_encoder_sentiment_xgb.pkl")

slang_dict = {}
with open(r"slang_dict_combined.txt", "r", encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split(",")
        if len(parts) == 2:
            slang, formal = parts
            slang_dict[slang.strip()] = formal.strip()

def preprocess_text(text):
    text = str(text).lower()
    text = emoji.demojize(text, delimiters=(" emoji_", " "))
    words = text.split()
    words = [slang_dict.get(w, w) for w in words]
    text = " ".join(words)
    text = re.sub(r"http\S+|www\S+|@\w+|#\w+", " ", text)
    text = re.sub(r"[^a-z0-9_\s]", " ", text)
    text = re.sub(r"\b\d+\b", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ======================================================
# 🔁 KOMENTAR: DARI DB LOKAL + FALLBACK SCRAPE (CACHED)
# ======================================================
def extract_reel_id(url: str) -> str | None:
    m = re.search(r"/reel/([^/?]+)", str(url))
    return m.group(1) if m else None

# =======================
# 🔧 Helpers (letakkan di atas get_comments_for_reel_id)
# =======================
@st.cache_data(show_spinner=False)
def get_comments_for_reel_id(reel_id: str) -> pd.DataFrame:
    try:
        st.write(f"🔍 Mengambil komentar untuk Reel ID: {reel_id}")
        excel_candidates = [
            "Data Komentar.xlsx",
            "./Data Komentar.xlsx",
            "instagram_comments/Data Komentar.xlsx",
            "./instagram_comments/Data Komentar.xlsx",
            "data/Data Komentar.xlsx",
            "./data/Data Komentar.xlsx",
        ]

        excel_path = next((p for p in excel_candidates if os.path.exists(p)), None)

        if excel_path:
            df = pd.read_excel(excel_path)

            url_col = next((c for c in df.columns if c.strip().lower() in ["url", "link"]), None)
            cmt_col = next((c for c in df.columns if c.strip().lower() in ["comment", "komentar"]), None)

            if url_col and cmt_col:
                df[url_col] = df[url_col].astype(str)

                # regex fleksibel: dukung /username/reel/ID dan /reel/ID
                pattern = r"(?:instagram\.com/)(?:[\w.-]+/)?reel/([A-Za-z0-9_-]+)"
                df["reel_id"] = df[url_col].apply(
                    lambda x: re.search(pattern, x).group(1)
                    if re.search(pattern, x)
                    else None
                )

                subset = df[df["reel_id"] == reel_id].copy()
                if not subset.empty:
                    comments = (
                        subset[[cmt_col]]
                        .rename(columns={cmt_col: "Comment"})
                        .dropna()
                        .astype(str)
                        .assign(Comment=lambda s: s["Comment"].str.strip())
                        .query("Comment != ''")
                        .drop_duplicates()
                    )
                    st.success(f"✅ Data komentar berhasil diambil ({len(comments)} komentar).")
                    return comments

        # ==== 2️⃣ FALLBACK OTOMATIS (MULTI-TOKEN) ====
        st.info("🔄 Mencoba kembali mengambil data komentar...")

        if ApifyClient is not None and APIFY_TOKENS:
            for i, token in enumerate(APIFY_TOKENS, start=1):
                try:
                    client = ApifyClient(token)
                    run_input = {
                        "directUrls": [f"https://www.instagram.com/reel/{reel_id}"],
                        "resultsLimit": 300
                    }
                    run = client.actor("SbK00X0JYCPblD2wp").call(run_input=run_input)

                    comments = []
                    for item in client.dataset(run["defaultDatasetId"]).iterate_items():
                        txt = str(item.get("text", "")).strip()
                        if txt:
                            comments.append(txt)

                    comments = list(dict.fromkeys(comments))  
                    if comments:
                        st.success(f"✅ Data komentar berhasil diambil ({len(comments)} komentar).")
                        return pd.DataFrame({"Comment": comments})

                except Exception as e:
                    continue

        st.error("❌ Tidak ada komentar yang dapat diambil.")
        return pd.DataFrame(columns=["Comment"])

    except Exception as e:
        print(f"[ERROR] Gagal mengambil komentar: {e}")
        st.error("⚠️ Terjadi kesalahan saat mengambil komentar.")
        return pd.DataFrame(columns=["Comment"])

# ======================================================
# 🧠 ANALISIS SENTIMEN & ASPEK DARI DF KOMENTAR
# ======================================================
def analyze_sentiment_from_df(df_comments: pd.DataFrame) -> pd.DataFrame:
    if df_comments is None or df_comments.empty:
        return pd.DataFrame()
    texts = df_comments["Comment"].fillna("").astype(str)
    texts = texts[texts.str.strip() != ""]
    if texts.empty:
        return pd.DataFrame()
    vec = tfidf.transform(texts.apply(preprocess_text))
    preds = xgb.predict(vec)
    probas = xgb.predict_proba(vec)
    sentiments = lbl.inverse_transform(preds)
    out = pd.DataFrame({
        "Comment": texts.values,
        "sentiment": sentiments,
        "confidence": np.max(probas, axis=1)
    })
    return out.drop_duplicates(subset="Comment").reset_index(drop=True)

@st.cache_resource(show_spinner=False)
def load_aspect_model():
    tfidf_aspek = joblib.load(r"tfidf_aspek.joblib")
    xgb_aspek = joblib.load(r"xgb_aspek.joblib")
    lbl_aspek = joblib.load(r"label_encoder_aspek_xgb.pkl")
    return tfidf_aspek, xgb_aspek, lbl_aspek

def analyze_aspect_from_df(df_comments: pd.DataFrame) -> pd.DataFrame:
    if df_comments is None or df_comments.empty:
        return pd.DataFrame()
    texts = df_comments["Comment"].fillna("").astype(str)
    texts = texts[texts.str.strip() != ""]
    if texts.empty:
        return pd.DataFrame()
    tfidf_aspek, xgb_aspek, lbl_aspek = load_aspect_model()
    vec = tfidf_aspek.transform(texts.apply(preprocess_text))
    preds = xgb_aspek.predict(vec)
    probas = xgb_aspek.predict_proba(vec)
    aspects = lbl_aspek.inverse_transform(preds)
    out = pd.DataFrame({
        "Comment": texts.values,
        "pred_aspect": aspects,
        "probabilities": np.max(probas, axis=1)
    })
    return out.drop_duplicates(subset="Comment").reset_index(drop=True)

# ======================================================
# 🧩 SUMOPOD INSIGHT GENERATOR
# ======================================================
def generate_summary(df):
    client = OpenAI(
        api_key=OPENAI_API_KEY,
        base_url="https://ai.sumopod.com/v1"
    )
    row = df.iloc[0]
    fullname = row.get("fullname", "Tidak diketahui")
    username = row.get("username", "")
    caption = row.get("caption", "")
    transcript = row.get("transcript", "")
    duration = row.get("duration", "Tidak diketahui")
    plays = row.get("plays", "0")
    views = row.get("views", "0")
    likes = row.get("likes", "0")
    comments = row.get("comments", "0")
    engagement = row.get("engagement", "0%")
    pos = row.get("jumlah_komentar_positif", 0)
    neu = row.get("jumlah_komentar_netral", 0)
    neg = row.get("jumlah_komentar_negatif", 0)
    ket = row.get("keterangan", "")

    # NB: hindari ekspresi dengan backslash di { } f-string
    transcript_text = transcript if transcript else "Tidak tersedia"
    prompt = f"""
Kamu adalah analis media sosial profesional yang bertugas menulis interpretasi menyeluruh terhadap video Instagram Reels secara objektif dan berbasis data.

Langkah kerja:

1️⃣ Tentukan dua hal utama terlebih dahulu:
   - Kategori video (pilih satu dari daftar berikut):
     ["Motivasi", "Berita", "Hiburan", "Opini", "Kegiatan Sosial", "Gaya Hidup", "Promosi Produk", "Musik", "Kuliner", "Otomotif", "Lain-lain"]
   - Klasifikasi: "ADS" (iklan/promosi) atau "NON-ADS" (bukan iklan/promosi)
   - Sertakan tingkat keyakinan (confidence level dalam %) untuk keduanya di bagian paling atas dengan format:
     Kategori: <nama kategori> (Confidence: <persentase>%)
     Klasifikasi: <ADS/NON-ADS> (Confidence: <persentase>%)

2️⃣ Setelah itu, tuliskan **dua paragraf** deskriptif:
   - **Paragraf pertama:**
     Gambarkan isi video secara alami dan komprehensif seolah-olah kamu menjelaskannya kepada seseorang yang belum menontonnya.  
     Di awal paragraf, sebutkan kategori dan sifat video (misalnya: “Sebagai video motivasi yang bersifat non-iklan…” atau “Sebagai video promosi produk…”).  
     Fokus pada isi dan makna pesan tanpa menyebut sumber data atau durasi.  
     Tutup paragraf dengan kalimat kesimpulan yang merangkum pesan utama video.

   - **Paragraf kedua:**
     Evaluasi performa video berdasarkan data interaksi: jumlah penayangan, likes, komentar, dan engagement secara umum.  
     Jelaskan bagaimana persepsi audiens tercermin dari proporsi sentimen berikut:
       - Positif: {pos}%
       - Netral: {neu}%
       - Negatif: {neg}%  
     Tulis dengan kalimat natural (misalnya: “Mayoritas komentar bersentimen positif sekitar 70%...”).  
     Berikan saran atau evaluasi singkat yang relevan berdasarkan pola data tersebut, tanpa menambahkan hal yang tidak ada di data (misalnya: visual, ekspresi, gaya editing, dll).

Gunakan semua data di bawah ini untuk memahami konteks video, namun jangan menyebut sumbernya secara eksplisit:
- Uploader: {fullname} ({username})
- Caption: {caption}
- Transcript Audio: {transcript_text}
- Plays: {plays}
- Views: {views}
- Likes: {likes}
- Komentar: {comments}
- Engagement Rate: {engagement}
- Komentar Positif (%): {pos}
- Komentar Netral (%): {neu}
- Komentar Negatif (%): {neg}
- Keterangan: {ket}

Gunakan bahasa Indonesia formal yang alami, informatif, dan padat makna.
Total maksimal 400 kata.
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.65,
        max_tokens=400
    )
    return response.choices[0].message.content.strip()

import plotly.express as px
import streamlit.components.v1 as components
import html, math
from collections import Counter

def plot_top_words(df, aspect_label, color):
    all_text = " ".join(df["Comment"].astype(str).tolist()).lower()
    words = [w for w in re.findall(r"\b[a-zA-Z]{3,}\b", all_text)]
    counter = Counter(words)
    common_words = counter.most_common(5)
    if not common_words:
        st.info("Tidak cukup kata untuk menampilkan grafik.")
        return
    top_df = pd.DataFrame(common_words, columns=["Kata", "Frekuensi"])
    fig = px.bar(top_df, x="Frekuensi", y="Kata", orientation="h", color_discrete_sequence=[color])
    fig.update_layout(
        yaxis=dict(autorange="reversed"),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=30, b=10),
        height=220,
        font=dict(color="white"),
        title=dict(text=f"🔍 Kata Dominan — {aspect_label}", font=dict(size=14, color="white"))
    )
    st.plotly_chart(fig, use_container_width=True)

import streamlit.components.v1 as components

import streamlit.components.v1 as components

def run_analyzer_page():
    html_block = """
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap" rel="stylesheet">
    <style>
    .reeltalk-header {
        background: linear-gradient(135deg, #0F172A, #1E293B);
        padding: 26px 32px;
        border-radius: 16px;
        box-shadow: 0 3px 14px rgba(0,0,0,0.35);
        margin-top: 15px;
        margin-bottom: 25px;
        font-family: 'Inter', sans-serif;
        text-align: left;
        color: #E2E8F0; /* default text color */
    }
    .reeltalk-header h1 {
        font-size: 30px;
        font-weight: 800;
        color: #F8FAFC;
        margin-bottom: 6px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .reeltalk-header h3 {
        font-size: 15px;
        font-weight: 500;
        color: #FACC15;
        margin-bottom: 12px;
    }
    .reeltalk-header b {
        color: #F1F5F9; /* fix bold text jadi terang */
    }
    .reeltalk-header p {
        font-size: 14.5px;
        line-height: 1.7;
        color: #CBD5E1;
        text-align: justify;
        margin: 0;
        max-width: 92%;
    }
    .sidebar-tip {
        background: rgba(59,130,246,0.1);
        border: 1px solid rgba(59,130,246,0.25);
        color: #BFDBFE;
        font-size: 13.8px;
        padding: 12px 16px;
        margin-top: 16px;
        border-radius: 10px;
        line-height: 1.6;
        box-shadow: inset 0 0 12px rgba(59,130,246,0.15);
    }
    .sidebar-tip b { color: #E0F2FE; }
    .sidebar-tip span { color: #60A5FA; font-weight:600; }
    </style>

    <div class="reeltalk-header">
        <h1>🔍 ViralLens AI</h1>
        <h3>✨ Lensa Pintar untuk Melihat Potensi Viral Kontenmu</h3>

        <p><b>ViralLens AI</b> membantu kamu membaca performa video secara cepat dan cerdas —
        dari <b>analisis komentar</b> dan <b>emosi audiens</b> hingga <b>tren topik</b> serta <b>waktu unggah terbaik</b>.
        Aplikasi ini jadi panduan praktis untuk memahami faktor yang membuat konten berpotensi viral. 🚀</p>

        <p style="margin-top: 10px;">
        Dilengkapi dua fitur utama:
        <br>• <b>🎬 ReelTalk</b> — analisis mendalam komentar, aspek, performa, dan transkrip video Reels.
        <br>• <b>📊 Dashboard Looker</b> — pantau tren, engagement, dan jam unggah paling efektif.
        </p>

        <p style="margin-top: 10px;">
        Karena viral bukan kebetulan — tapi hasil dari memahami data dengan tepat. 💡
        </p>

        <div class="sidebar-tip">
            💡 <b>Menu navigasi tersedia di sidebar kiri.</b><br>
            Gunakan untuk <b>berpindah halaman</b> antara 
            <span>🎬 ReelTalk Analyzer</span> dan 
            <span>📊 Dashboard Looker</span>.
        </div>
    </div>
    """
    components.html(html_block, height=475, scrolling=False)


# ================================
# 🎥 Input URL Instagram Reels
# ================================
    st.markdown("""
    <div style='
        padding: 10px 14px;
        border-radius: 10px;
        background-color: rgba(241,245,249,0.08);
        border: 1px solid rgba(148,163,184,0.25);
    '>
        <h4 style='margin-bottom: 4px; color:#F8FAFC; font-size:17px;'>
            🎥 <b>Masukkan URL Instagram Reels</b>
        </h4>
        <p style='font-size:14.2px; line-height:1.55; color:#CBD5E1; margin-bottom:5px;'>
            Kamu bisa mencoba dengan salah satu <span style='color:#F1F5F9; font-weight:600;'>contoh video</span> di bawah ini,  
            atau <b>masukkan link Reels kamu sendiri</b>.
        </p>
        <p style='font-size:13.5px; color:#FCA5A5; margin-top:2px;'>
            ⚠️ <b>Catatan:</b> contoh ini hanya untuk <b>uji coba</b> agar kamu bisa melihat bagaimana hasil analisis tampil.  
            Semua video lain akan dianalisis dengan cara yang sama 👇
        </p>
    </div>
    """, unsafe_allow_html=True)

    
    # daftar contoh video real
    contoh_reel_links = {
        "📱 Contoh 1 — David Gadgetin (Review Tekno)": "https://www.instagram.com/reel/DHTC04Vybkk/?igsh=MXIzYmx6NXBzdzdqOQ%3D%3D",
        "🚗 Contoh 2 — Nexcarlos (Kuliner)": "https://www.instagram.com/reel/DMz1mj7s6u7/?igsh=MXQzN3ZoNWZsMGk5cg%3D%3D",
        "🏍 Contoh 3 — Fitra Eri (Otomotif)": "https://www.instagram.com/reel/DMEz84OyvC1/?igsh=bjA5dGVkeGtxMmM1",
    }
    
    col1, col2 = st.columns([1.8, 1.2])
    
    with col1:
        url = st.text_input(
            "Masukkan URL Instagram Reels:",
            key="url_input_main",
            placeholder="https://www.instagram.com/reel/XXXXX/",
        )
    
    with col2:
        selected_example = st.selectbox(
            "Atau pilih contoh video:",
            ["(Pilih salah satu contoh)"] + list(contoh_reel_links.keys()),
            key="example_selector",
        )
    
    # jika user pilih contoh, isi otomatis field input-nya
    if selected_example != "(Pilih salah satu contoh)":
        url = contoh_reel_links[selected_example]
        st.info(f"🔗 Menggunakan contoh: **{selected_example}**")
    
    # tombol jalankan analisis
    if st.button("🚀 Jalankan Analisis Lengkap", key="run_btn"):
        # cocokkan pola URL Reels (dengan atau tanpa username)
        valid_url = re.search(r"(?:instagram\.com/)(?:[\w.-]+/)?reel/([A-Za-z0-9_-]+)", url)
    
        if not valid_url:
            st.error("❌ URL tidak valid. Pastikan mengandung '/reel/<ID>', misalnya:\n"
                     "- https://www.instagram.com/reel/XXXXX/\n"
                     "- https://www.instagram.com/<username>/reel/XXXXX/")
        else:
            # clear state untuk run baru
            for k in list(st.session_state.keys()):
                if k not in ["url_input_main", "nav_radio", "example_selector"]:
                    del st.session_state[k]
            st.session_state["run_new_analysis"] = True
            st.rerun()



    if st.session_state.get("run_new_analysis", False) and "analysis_data" not in st.session_state:
        with st.status("🚀 Menjalankan analisis lengkap...", expanded=True) as status:
            st.write("⏳ Mengambil data metrik...")
            data = scrape_instagram_reel(url)
            if not data:
                status.update(label="❌ Gagal mengambil metrik.", state="error")
                st.stop()

            st.write("🌐 Cek Video ...")
            audio_url = get_audio_from_instagram(url)
            if not audio_url:
                status.update(label="❌ Gagal ambil audio.", state="error")
                st.stop()

            st.write("📥 Mengunduh file audio...")
            tmp_audio = download_file(audio_url, ".mp3")

            st.write("🧠 Mentranskripsi audio...")
            transcript = transcribe_audio(tmp_audio)

            # === NEW: ambil komentar (DB lokal -> fallback scrape jika diset) ===
            st.write("💬 Mengambil komentar...")
            comments_df = get_comments_for_reel_id(data["reel_id"])

            # === NEW: analisis sentiment/aspek dari DF yang sama ===
            if comments_df.empty:
                pos = neu = neg = None
                ket = "Tidak ada komentar / akses dibatasi IG"
                subset = pd.DataFrame()
                aspect_df = pd.DataFrame()
            else:
                subset = analyze_sentiment_from_df(comments_df)
                total = len(subset) if not subset.empty else 0
                if total > 0:
                    pos = round((subset["sentiment"] == "positive").sum() / total * 100, 2)
                    neu = round((subset["sentiment"] == "neutral").sum() / total * 100, 2)
                    neg = round((subset["sentiment"] == "negative").sum() / total * 100, 2)
                    pos, neu, neg = f"{pos}%", f"{neu}%", f"{neg}%"
                    ket = "Ada komentar"
                else:
                    pos = neu = neg = None
                    ket = "Tidak ada komentar / akses dibatasi IG"

                aspect_df = analyze_aspect_from_df(comments_df)

            status.update(label="✅ Analisis lengkap selesai!", state="complete")

        st.session_state["analysis_data"] = {
            "data": data,
            "transcript": transcript,
            "pos": pos, "neu": neu, "neg": neg, "ket": ket,
            "subset": subset,
            "aspect_df": aspect_df
        }
        st.session_state["run_new_analysis"] = False
        st.rerun()

    if "analysis_data" not in st.session_state:
        return

    saved = st.session_state["analysis_data"]
    data = saved["data"]; transcript = saved["transcript"]
    pos, neu, neg, ket = saved["pos"], saved["neu"], saved["neg"], saved["ket"]
    subset = saved["subset"]
    aspect_df = saved.get("aspect_df", pd.DataFrame())

    st.success("✅ Analisis lengkap tersedia!")
    df = pd.DataFrame([{
        **data, "transcript": transcript,
        "jumlah_komentar_positif": pos,
        "jumlah_komentar_netral": neu,
        "jumlah_komentar_negatif": neg,
        "keterangan": ket
    }])[["fullname","username","caption","duration","plays","views","likes","comments","engagement",
         "transcript","jumlah_komentar_positif","jumlah_komentar_netral","jumlah_komentar_negatif","keterangan"]]
    st.dataframe(df, use_container_width=True)

    if "summary" not in st.session_state:
        with st.spinner("🧠 Menghasilkan ringkasan insight..."):
            st.session_state["summary"] = generate_summary(df)

    # ----- Ringkasan Insight -----
    st.markdown("### 📋 Ringkasan Insight Video")
    summary_html_text = st.session_state["summary"].replace("\n", "<br>")
    st.markdown(
        f"""
        <div style="
            background-color: rgba(240, 242, 246, 0.1);
            border: 1px solid rgba(120, 120, 120, 0.4);
            border-radius: 12px;
            padding: 18px 22px;
            margin-top: 8px;
            margin-bottom: 25px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.2);
        ">
            <p style="font-size: 15px; line-height: 1.7; text-align: justify;">
                {summary_html_text}
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Jika komentar 0 → skip semua section di bawah (hindari KeyError)
    if int(data.get("comments") or 0) == 0 or subset is None or subset.empty:
        st.info("💬 Video ini tidak memiliki komentar publik untuk dianalisis / akses dibatasi.")
        return

    # ---------- SENTIMEN ----------
    if not subset.empty:
        st.markdown("## 💬 Analisis Komentar Berdasarkan Sentimen")

        sentiment_counts = {
            "Positif": (subset["sentiment"] == "positive").sum(),
            "Netral": (subset["sentiment"] == "neutral").sum(),
            "Negatif": (subset["sentiment"] == "negative").sum()
        }
        pie_df = pd.DataFrame({"Sentimen": list(sentiment_counts.keys()), "Jumlah": list(sentiment_counts.values())})
        fig_pie = px.pie(
            pie_df, names="Sentimen", values="Jumlah",
            color="Sentimen",
            color_discrete_sequence=["#2ECC71", "#74B9FF", "#E17055"],
            hole=0.5
        )
        fig_pie.update_traces(textinfo="percent+label", textfont_size=14)
        fig_pie.update_layout(
            title_text="📊 Distribusi Sentimen Komentar",
            title_font=dict(size=20, color="white"),
            legend=dict(orientation="h", y=-0.2, x=0.5, xanchor="center", font=dict(size=14, color="white")),
            margin=dict(t=30, b=0, l=0, r=0),
            height=280
        )
        st.plotly_chart(fig_pie, use_container_width=True)

        sentiment_option = st.selectbox(
            "Pilih tampilan sentimen:",
            ["Semua", "Positif", "Netral", "Negatif"],
            index=0,
            key="sentiment_filter_main"
        )
        if "last_sentiment_option" not in st.session_state or st.session_state["last_sentiment_option"] != sentiment_option:
            st.session_state["last_sentiment_option"] = sentiment_option
            for k in list(st.session_state.keys()):
                if k.startswith("page_sentiment_"):
                    del st.session_state[k]

        sentiment_map = {"Positif": "positive", "Netral": "neutral", "Negatif": "negative"}
        sentiment_colors = {"positive": "#2ECC71", "neutral": "#0984E3", "negative": "#E17055"}
        gradient_colors = {
            "positive": "linear-gradient(135deg, #00C851, #009245, #004D40)",
            "neutral": "linear-gradient(135deg, #74B9FF, #0984E3, #002B5B)",
            "negative": "linear-gradient(135deg, #FF6B6B, #C0392B, #5A0000)"
        }
        sentiment_desc = {
            "positive": "💚 Komentar yang bernada positif, menunjukkan dukungan, kebanggaan, atau apresiasi terhadap video.",
            "neutral": "💙 Komentar netral/informatif tanpa emosi kuat.",
            "negative": "❤️ Komentar berisi kritik atau ketidakpuasan."
        }
        selected_sents = ["positive", "neutral", "negative"] if sentiment_option == "Semua" else [sentiment_map[sentiment_option]]
        font_style = "font-family: 'Poppins', sans-serif; letter-spacing: 0.3px;"

        import html as ihtml
        for senti in selected_sents:
            senti_df = subset[subset["sentiment"] == senti].copy()
            if senti_df.empty:
                continue
            color = sentiment_colors[senti]
            gradient = gradient_colors[senti]
            label = senti.capitalize()
            desc = sentiment_desc[senti]

            st.markdown(f"""
            <div style="border-radius:14px;padding:12px 18px;margin:28px 0 18px 0;background:{gradient};
                        box-shadow:0 0 15px rgba(0,0,0,0.35);backdrop-filter:blur(6px);-webkit-backdrop-filter:blur(6px);">
              <h5 style="margin:0;font-size:16px;font-weight:600;color:white;{font_style}">
                🎯 {label} Sentiment — <span style='opacity:.9'>{len(senti_df.drop_duplicates(subset="Comment")):,} komentar</span>
              </h5>
              <p style="margin-top:6px;font-size:13.5px;color:rgba(255,255,255,.9);{font_style}">{desc}</p>
            </div>
            """, unsafe_allow_html=True)

            c1, c2 = st.columns([1, 1.2])
            with c1:
                all_text = " ".join(senti_df["Comment"].astype(str).tolist())
                if all_text.strip():
                    wc = WordCloud(width=600, height=300, background_color="white", colormap="Set2").generate(all_text)
                    fig, ax = plt.subplots(figsize=(5, 3))
                    ax.imshow(wc, interpolation="bilinear")
                    ax.axis("off")
                    st.pyplot(fig)
                else:
                    st.info("Tidak cukup teks untuk membuat WordCloud.")

            with c2:
                senti_df = senti_df.sort_values("confidence", ascending=False).drop_duplicates(subset="Comment").reset_index(drop=True)
                total = len(senti_df)
                per_page = 50
                pages = max(1, math.ceil(total / per_page))
                page_num = st.number_input(f"Halaman komentar ({label})", 1, pages, 1, key=f"page_sentiment_{label}")
                start, end = (page_num - 1) * per_page, min((page_num) * per_page, total)

                cards = ""
                for _, row in senti_df.iloc[start:end].iterrows():
                    comment_text = str(row.get("Comment", "")).strip()
                    if not comment_text:
                        continue
                    conf = round(float(row.get("confidence", 0)) * 100, 2)
                    safe_comment = ihtml.escape(comment_text).replace("&lt;br&gt;", "<br>").replace("&lt;br /&gt;", "<br>")
                    cards += f"""
                    <div style="border-left:4px solid {color};padding:12px 16px;margin-bottom:10px;background:rgba(255,255,255,.08);
                                border-radius:14px;backdrop-filter:blur(8px);-webkit-backdrop-filter:blur(8px);
                                box-shadow:0 3px 10px rgba(0,0,0,.25);transition:all .25s;transform:translateY(0)">
                      <p style="margin:0;font-size:14.5px;line-height:1.6;color:white;{font_style}">💬 {safe_comment}</p>
                      <p style="margin:4px 0 0 0;color:#B0B0B0;font-size:12.5px;{font_style}">🔹 Keyakinan: {conf}%</p>
                    </div>"""

                if cards.strip():
                    components.html(f"""
                      <div style="height:400px;padding:12px;border-radius:14px;background:rgba(255,255,255,.05);
                                  box-shadow:inset 0 0 12px rgba(255,255,255,.05), 0 4px 10px rgba(0,0,0,.3);
                                  backdrop-filter:blur(8px);-webkit-backdrop-filter:blur(8px);overflow-y:auto;color:white;{font_style}">
                        {cards}
                      </div>
                    """, height=420, scrolling=False)
                    st.caption(f"Menampilkan komentar {start+1:,}–{end:,} dari {total:,}")
                else:
                    st.info("⚠️ Tidak ada komentar pada halaman ini.")

    # ---------- ASPEK ----------
    if not aspect_df.empty:
        st.markdown("---")
        st.markdown("## 🧩 Analisis Komentar Berdasarkan Aspek")
        st.caption("📍 Komentar dikelompokkan berdasarkan konteks pembahasan pengguna terhadap video.")

        all_labels = ["Semua"] + sorted(aspect_df["pred_aspect"].unique().tolist())
        selected_aspect = st.selectbox(
            "Pilih aspek yang ingin ditampilkan:",
            all_labels,
            index=0,
            key="aspect_filter_main"
        )
        if "last_aspect_option" not in st.session_state or st.session_state["last_aspect_option"] != selected_aspect:
            st.session_state["last_aspect_option"] = selected_aspect
            for k in list(st.session_state.keys()):
                if k.startswith("page_aspect_"):
                    del st.session_state[k]

        display_df = aspect_df if selected_aspect == "Semua" else aspect_df[aspect_df["pred_aspect"] == selected_aspect]

        aspect_colors = {
            "Pujian": "#2ecc71",
            "Perbandingan / Saran / Ide": "#2980b9",
            "Pertanyaan / Request": "#8e44ad",
            "Pengalaman / Hasil": "#16a085",
            "Keluhan / Kesulitan": "#e67e22",
            "Lainnya / Tidak Relevan": "#7f8c8d"
        }
        aspect_desc = {
            "Pujian": "💚 Komentar berisi apresiasi/kebanggaan.",
            "Perbandingan / Saran / Ide": "💡 Komentar memberi perbandingan, masukan, atau ide.",
            "Pertanyaan / Request": "❓ Komentar berupa pertanyaan/permintaan info.",
            "Pengalaman / Hasil": "🧠 Komentar membagikan pengalaman/hasil.",
            "Keluhan / Kesulitan": "⚠️ Komentar menyampaikan kritik/kendala.",
            "Lainnya / Tidak Relevan": "🌀 Komentar umum/tidak terkait langsung."
        }
        custom_order = [
            "Pujian",
            "Perbandingan / Saran / Ide",
            "Pertanyaan / Request",
            "Pengalaman / Hasil",
            "Keluhan / Kesulitan",
            "Lainnya / Tidak Relevan"
        ]
        aspects_order = sorted(
            display_df["pred_aspect"].unique(),
            key=lambda x: custom_order.index(x) if x in custom_order else len(custom_order)
        )

        import html as ihtml2
        for aspect in aspects_order:
            asp_df = display_df[display_df["pred_aspect"] == aspect].copy()
            asp_df = asp_df.sort_values("probabilities", ascending=False).drop_duplicates(subset="Comment").reset_index(drop=True)

            color = aspect_colors.get(aspect, "#3498db")
            desc = aspect_desc.get(aspect, "")

            st.markdown(f"""
            <div style="border-radius:12px;padding:12px 16px;margin:30px 0 15px 0;
                        background: linear-gradient(135deg, {color}, rgba(255,255,255,0.08));
                        box-shadow:0 0 12px rgba(0,0,0,0.25);backdrop-filter:blur(6px);-webkit-backdrop-filter:blur(6px);">
            <h5 style="margin:0;font-size:16px;font-weight:600;color:white;font-family:'Poppins',sans-serif;">
                🎯 Aspek: {aspect} — {len(asp_df):,} komentar
            </h5>
            <p style="margin-top:6px;font-size:13.5px;color:rgba(255,255,255,.85);font-family:'Poppins',sans-serif;">{desc}</p>
            </div>
            """, unsafe_allow_html=True)

            c1, c2 = st.columns([1, 1.2])
            with c1:
                plot_top_words(asp_df, aspect, color)

            with c2:
                total = len(asp_df)
                per_page = 50
                pages = max(1, math.ceil(total / per_page))
                page_num = st.number_input(
                    f"Halaman komentar ({aspect})",
                    1, pages, 1,
                    key=f"page_aspect_{aspect}"
                )
                start, end = (page_num - 1) * per_page, min(page_num * per_page, total)

                cards = ""
                for _, row in asp_df.iloc[start:end].iterrows():
                    txt = str(row.get("Comment", "")).strip()
                    if not txt:
                        continue
                    prob = round(float(row.get("probabilities", 0)) * 100, 2)
                    safe = ihtml2.escape(txt)
                    cards += f"""
                    <div style="border-left:4px solid {color};padding:12px 16px;margin-bottom:10px;background:rgba(255,255,255,.08);
                                border-radius:14px;backdrop-filter:blur(8px);-webkit-backdrop-filter:blur(8px);
                                box-shadow:0 3px 10px rgba(0,0,0,.25);transition:all .25s;transform:translateY(0)">
                    <p style="margin:0;font-size:14.5px;line-height:1.6;color:white;font-family:'Poppins',sans-serif;">
                        💬 {safe}
                    </p>
                    <p style="margin:4px 0 0 0;color:#B0B0B0;font-size:12.5px;">🔹 Keyakinan: {prob}%</p>
                    </div>"""

                if cards.strip():
                    components.html(f"""
                    <div style="height:400px;padding:12px;border-radius:14px;background:rgba(255,255,255,.05);
                                box-shadow:inset 0 0 12px rgba(255,255,255,.05), 0 4px 10px rgba(0,0,0,.3);
                                backdrop-filter:blur(8px);-webkit-backdrop-filter:blur(8px);
                                overflow-y:auto;color:white;font-family:'Poppins',sans-serif;">
                    {cards}
                    </div>
                    """, height=420, scrolling=False)
                    st.caption(f"Menampilkan komentar {start+1:,}–{end:,} dari {total:,}")
                else:
                    st.info("⚠️ Tidak ada komentar pada halaman ini.")

# ======================================================
# PAGE: LOOKER
# ======================================================
import streamlit as st

def run_looker_page():
    # === HEADER ===
    st.markdown("""
    <h1 style="font-weight:800; font-size:30px; margin-bottom:6px;">
        📈 ViralLens Dashboard — Social Media Analytics
    </h1>
    <p style="color:#CBD5E1; font-size:14.5px; line-height:1.6; margin-bottom:20px;">
        Pantau performa dan perilaku audiens Instagram Reels secara interaktif.  
        Tutup sidebar agar dashboard tampil lebih luas dan proporsional.
    </p>
    """, unsafe_allow_html=True)

    # === 2 KOLOM UNTUK DROPDOWN DAN RADIO ===
    col1, col2 = st.columns([1.6, 1.2])

    with col1:
        page_option = st.selectbox(
            "Halaman Dashboard:",
            [
                "Halaman 1 – Executive Summary & Overview",
                "Halaman 2 – Content Characteristics & Engagement",
                "Halaman 3 – When to Post (Timing & Audience Behavior)",
                "Halaman 4 – Creator Leaderboard & Benchmark"
            ],
            label_visibility="visible"
        )

    with col2:
        view_mode = st.radio(
            "Tampilan:",
            ["Sidebar Terbuka", "Sidebar Tertutup"],
            horizontal=True,
            label_visibility="visible"
        )
        iframe_height = 800 if "Terbuka" in view_mode else 1000

    # === INFO CARD DENGAN AKSEN MERAH ===
    st.markdown("""
    <div style="
        background: rgba(59, 130, 246, 0.15);
        border-left: 4px solid #3b82f6;
        border-radius: 6px;
        padding: 10px 14px;
        margin-top: 10px;
        margin-bottom: 18px;
        font-size: 14px;
        color: #bfdbfe;
        line-height: 1.5;">
        💡 <b>Pilih halaman</b> untuk menampilkan penjelasan di bawah.<br>
        🌐 Anda juga dapat berinteraksi langsung dengan dashboard — misalnya menggunakan filter kategori, topik, atau kreator.<br>
        📘 Anda bisa berpindah halaman dengan klik halaman dari dashboard, namun untuk menampilkan penjelasan di bawah, tetap pilih halaman melalui menu di atas.
    </div>
    """, unsafe_allow_html=True)

    # === URL LOOKER ===
    base_url = "https://lookerstudio.google.com/embed/reporting/e3b84c42-6761-438e-9bed-5940ecee866c"
    looker_pages = {
        "Halaman 1 – Executive Summary & Overview": f"{base_url}/page/p_07prsgozwd",
        "Halaman 2 – Content Characteristics & Engagement": f"{base_url}/page/p_pbsyk8g0wd",
        "Halaman 3 – When to Post (Timing & Audience Behavior)": f"{base_url}/page/p_e3p3l8g0wd",
        "Halaman 4 – Creator Leaderboard & Benchmark": f"{base_url}/page/p_qshnx7g0wd"
    }

    # === EMBED LOOKER ===
    st.markdown(f"""
        <iframe id="looker-frame"
            src="{looker_pages[page_option]}"
            style="border:0;width:100%;height:{iframe_height}px;
                   border-radius:12px;box-shadow:0 0 15px rgba(0,0,0,0.25);
                   transition:height 0.4s ease;"
            allowfullscreen>
        </iframe>
    """, unsafe_allow_html=True)

    st.caption(f"Tampilan saat ini: **{view_mode}** — sesuaikan bila ukuran tampilan tidak proporsional.")
    st.markdown("---")
    # ======================================================
    # 🧭 PENJELASAN PER HALAMAN
    # ======================================================
    if page_option == "Halaman 1 – Executive Summary & Overview":
        st.subheader("🧭 Executive Summary & Overview")
        st.markdown("""
Dashboard ini menampilkan **gambaran umum performa konten Instagram** tanpa filter kategori, topik, atau tipe konten.  
Data di sini menjadi **baseline awal** untuk memahami perilaku audiens dan efektivitas strategi konten.

**Insight utama:**
- Rata-rata performa konten: **Plays 756K**, **Views 153.9K**, **Likes 17.9K**, **Comments 264**, **ER 2.92%**.  
- **Kategori engagement tertinggi:** Edukasi (7%), Otomotif (6%), Gaya Hidup (5%).  
- **Topik populer:** Edukasi (22.5%), Otomotif (15.5%), Lifestyle (14.9%).  
- **Hari unggah terbaik:** Rabu & Kamis; **Non-Ads** unggul dibanding Ads.  
- **Kreator menonjol:** `@fanzoneid_official`, `@hybicara`, `@piala_pertiwi`.

**Kesimpulan:**  
Konten edukatif & lifestyle dengan gaya natural dan non-iklan menjadi magnet utama audiens.

💡 **Saran untuk pengguna:**
1. Jadikan ini **patokan awal** performa akunmu dibanding industri.  
2. Prioritaskan **konten edukatif dan lifestyle** untuk menjangkau audiens luas.  
3. Gunakan kategori dengan ER tinggi sebagai inspirasi ide konten berikutnya.
        """)

    elif page_option == "Halaman 2 – Content Characteristics & Engagement":
        st.subheader("🎞️ Content Characteristics & Engagement")
        st.markdown("""
Halaman ini menjelaskan bagaimana **durasi video dan jenis konten (Ads/Non-Ads)** memengaruhi engagement dan rewatch rate.

**Insight utama:**
- **Non-Ads** punya engagement lebih tinggi dibanding Ads.  
- **Durasi efektif:** pendek–menengah (≤60 detik) → ER 20–30%.  
- **Kategori dengan total plays tertinggi:**  
  🎨 *Makeup Artist / Beauty Influencer* (130.5 jt)  
  💄 *Product Review / Skincare* (42.3 jt)  
  🧴 *Skincare / Beauty Content* (32.4 jt)  
  🚗 *Car Enthusiast / YouTube Channel* (29.7 jt)  
- **Top performer:**  
  `@fitra.eri` – 26.13% (10 detik)  
  `@mercia_review` – 22.92% (1 menit)  
  `@fajaralfian95` – 16%  
  `@ibnujamilo` – 12%

**Kesimpulan:**  
Reels pendek–menengah dengan pesan ringan & edukatif memberikan engagement tertinggi.  
Video panjang punya views tinggi tapi interaksi rendah.

💡 **Saran untuk pengguna:**
1. Gunakan durasi **15–45 detik** untuk hasil optimal.  
2. Buat konten dengan **storytelling ringan & natural**.  
3. Untuk brand, fokus ke durasi **30–60 detik** dengan visual kuat.  
4. Jadikan konten **rewatch tinggi** sebagai template untuk seri berikutnya.
        """)

    elif page_option == "Halaman 3 – When to Post (Timing & Audience Behavior)":
        st.subheader("⏰ When to Post (Timing & Audience Behavior)")
        st.markdown("""
Halaman ini membantu menemukan **hari & jam unggah paling efektif** untuk engagement dan likes tertinggi.

**Insight utama:**
- **Waktu unggah optimal:**  
  🌙 20:00–23:59 → prime time malam  
  🌅 08:00–11:59 → engagement stabil  
  ☀️ 16:00–19:59 → sore hari efektif  
  🕛 00:00–03:59 (Senin dini hari) → engagement tertinggi (0.05) karena kompetisi rendah.  
- **Hari terbaik:** Jumat, Selasa, Rabu, & Senin dini hari.  
- **Top Reels & waktu posting:**  
  - `@fitra.eri` – Jumat siang (26.13%)  
  - `@mercia_review` – Sabtu pagi (22.92%)  
  - `@fajaralfian95` – Selasa malam (16%)  
  - `@ibnujamilo` – Rabu pagi (12.14%)  

**Kesimpulan:**  
Engagement tertinggi terjadi di **malam hari & pertengahan minggu**, dengan **Senin dini hari** sebagai waktu kejutan potensial.

💡 **Saran untuk pengguna:**
1. Posting utama di **20.00–23.00 (Selasa–Jumat)**.  
2. Tes **Senin dini hari (00.00–03.59)** untuk soft-launch atau niche.  
3. Hindari unggahan jam kerja (08.00–15.00).  
4. Gunakan **scheduler otomatis** agar unggahan muncul di waktu ideal.
        """)

    elif page_option == "Halaman 4 – Creator Leaderboard & Benchmark":
        st.subheader("👑 Creator Leaderboard & Benchmark")
        st.markdown("""
Halaman ini membandingkan **performa antar kreator** berdasarkan followers, engagement profile, dan rewatch rate.  
Membantu kamu menemukan kreator paling efektif dan memahami pola kolaborasi ideal.

**Insight utama:**
- **Top akun berdasarkan followers:**  
  `@hudabeauty` (57M), `@ivan_gunawan` (31M), `@hotmanparisofficial` (10M), `@erickthohir` (8M), `@aniesbaswedan` (8M).  
- **Engagement profile tertinggi:**  
  `@fanzoneid_official` (1.25), `@hybicara` (0.78), `@piala_pertiwi` (0.64).  
- **Top creators by ER:**  
  `@fitra.eri` – 4.78% (Rewatch 144.6%)  
  `@ikhsanqthi` – 6.03% (Rewatch 2,055%)  
  `@moladin_id` – 2.18% (Rewatch 6,962%)  
- **Distribusi followers:**  
  <100K (48.9%), 100K–1M (33.8%), 1M–10M (14.7%).  
  → Micro & mid-tier influencer mendominasi engagement tinggi.

**Kesimpulan:**  
Kreator besar belum tentu paling efektif.  
**Micro & mid-tier influencer** justru punya audiens lebih loyal & responsif.

💡 **Saran untuk pengguna:**
1. Kolaborasi dengan kreator **10K–1M followers** untuk engagement tinggi.  
2. Gunakan *engagement profile* & *rewatch rate* sebagai indikator utama.  
3. Pilih kreator relevan dalam kategori/topik serupa.  
4. Pantau tren tiap tier untuk strategi kolaborasi jangka panjang.
        """)


# ======================================================
# SIDEBAR NAV
# ======================================================
# ===== Sidebar Styling =====
st.markdown("""
<style>
/* ---------- Sidebar Layout ---------- */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #111827, #1E293B);
    color: #E2E8F0;
    font-family: 'Inter', sans-serif;
    padding-top: 20px;
    border-right: 1px solid rgba(255,255,255,0.05);
}

/* ---------- Titles ---------- */
.sidebar-title {
    font-size: 18px;
    font-weight: 700;
    color: #F8FAFC;
    margin-bottom: 8px;
}

.sidebar-subtitle {
    font-size: 13.5px;
    color: #A5B4FC;
    margin-bottom: 14px;
}

/* ---------- Feature List ---------- */
.sidebar-feature {
    font-size: 13.5px;
    color: #CBD5E1;
    line-height: 1.65;
    margin-left: 8px;
}

.sidebar-feature li {
    margin-bottom: 4px;
}

.sidebar-feature li::marker {
    color: #818CF8;
}

/* ---------- Divider ---------- */
.sidebar-divider {
    margin: 10px 0 18px 0;
    border: none;
    border-top: 1px solid rgba(255,255,255,0.08);
}

/* ---------- Expander ---------- */
.streamlit-expanderHeader {
    font-size: 13px !important;
    font-weight: 600 !important;
    color: #F1F5F9 !important;
}

.streamlit-expanderHeader:hover {
    color: #A5B4FC !important;
}
</style>
""", unsafe_allow_html=True)

# ===== Sidebar Content =====
# ===== Sidebar Navigation =====
st.sidebar.markdown('<p class="sidebar-title">Navigasi Aplikasi</p>', unsafe_allow_html=True)
page = st.sidebar.radio(
    "Pilih Halaman:",
    ["🎬 ReelTalk Analyzer", "📈 Dashboard Looker"],
    key="nav_radio"
)

st.sidebar.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)

st.sidebar.markdown('<p class="sidebar-title">🔍 ViralLens AI</p>', unsafe_allow_html=True)


st.sidebar.markdown("""
<ul class="sidebar-feature">
  <b>Fitur Utama:</b>
  <li>Analisis Sentimen & Emosi</li>
  <li>Klasifikasi Aspek Konten</li>
  <li>Summary Video</li>
  <li>Dashboard Analitik</li>
</ul>
""", unsafe_allow_html=True)


st.sidebar.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)

# ===== Utilities Section =====
with st.sidebar.expander("⚙️ Utilities", expanded=False):
    st.caption("Kelola cache agar hasil analisis tetap bersih dan akurat.")

    if st.button("🧹 Hapus cache komentar", key="clear_comment_cache_btn"):
        try:
            get_comments_for_reel_id.clear()
            st.success("Cache komentar berhasil dibersihkan.")
        except Exception:
            st.cache_data.clear()
            st.success("Cache data dibersihkan (fallback).")
        st.rerun()

    if st.button("🧽 Hapus SEMUA cache (data + resource)", key="clear_all_cache_btn"):
        try:
            st.cache_data.clear()
            st.cache_resource.clear()
            st.success("Semua cache berhasil dibersihkan.")
        finally:
            st.rerun()

# ===== Page Routing =====
if page == "🎬 ReelTalk Analyzer":
    run_analyzer_page()
else:

    run_looker_page()





















