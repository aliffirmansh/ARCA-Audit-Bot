import streamlit as st
import pandas as pd
import faiss
import re
import time
from sentence_transformers import SentenceTransformer
from groq import Groq

# ===================================================================
# KONFIGURASI HALAMAN
# ===================================================================
st.set_page_config(
    page_title="ARCA - Audit Regulation Chat Assistant",
    layout="centered",
    page_icon="⚖️",
    initial_sidebar_state="expanded"
)

# ===================================================================
# KONFIGURASI SISTEM
# Dataset  : Regulasi Only (pojk_regulasi_only)
# Threshold: 5.5 (Ketat) — hasil terbaik dari evaluasi penelitian
# LLM      : qwen/qwen3-32b
# ===================================================================
THRESHOLD   = 5.5
LLM_MODEL   = "qwen/qwen3-32b"
TOP_K       = 10
MAX_CONTEXT = 3

GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

# ===================================================================
# LOAD ASSETS (DENGAN CACHING)
# ===================================================================
@st.cache_resource
def load_arca_engine():
    """Memuat model embedding, FAISS index, dan dataframe ke memori."""
    model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
    index = faiss.read_index("data/pojk_regulasi_only.index")
    df    = pd.read_pickle("data/pojk_regulasi_only_df.pkl")
    client = Groq(api_key=GROQ_API_KEY)
    return model, index, df, client

with st.spinner("⏳ Menginisialisasi ARCA..."):
    model, index, df, client = load_arca_engine()

# ===================================================================
# FUNGSI RAG
# ===================================================================
def search_pojk(query, k=TOP_K):
    """
    Melakukan semantic search ke FAISS index.
    Mengembalikan dataframe hasil retrieval dengan kolom 'distance'.
    """
    query_vector = model.encode([query])
    D, I = index.search(query_vector, k)
    results_df = df.iloc[I[0]].copy()
    results_df['distance'] = D[0]
    return results_df

def retrieve_context(query):
    """
    Pipeline retrieval dengan threshold tunggal T5.5.
    Pertanyaan di luar POJK 11 akan menghasilkan DataFrame kosong.
    """
    raw_results   = search_pojk(query)
    relevant      = raw_results[raw_results['distance'] <= THRESHOLD]

    if relevant.empty:
        return pd.DataFrame(), raw_results

    final_context = relevant.head(MAX_CONTEXT)
    return final_context, raw_results

def generate_answer_stream(messages, context_list):
    """
    Mengirim konteks regulasi dan riwayat percakapan ke LLM via Groq API.
    Menggunakan streaming agar respons muncul bertahap.
    Menghapus tag <think> dari output model reasoning Qwen.
    """
    context_str = "\n\n---\n\n".join(context_list)

    system_prompt = {
        "role": "system",
        "content": (
            "Anda adalah ARCA (Audit Regulation Chat Assistant), asisten audit TI senior "
            "yang ahli dalam Peraturan Otoritas Jasa Keuangan (POJK) No. 11/POJK.03/2022 "
            "tentang Penyelenggaraan Teknologi Informasi oleh Bank Umum.\n\n"
            "ATURAN KETAT:\n"
            "1. Jawab HANYA berdasarkan KONTEKS REGULASI yang disediakan di bawah ini.\n"
            "2. Jika informasi tidak ada dalam konteks, nyatakan secara eksplisit: "
            "'Informasi tersebut tidak tersedia dalam regulasi yang dirujuk.'\n"
            "3. Selalu sebutkan pasal yang menjadi dasar jawaban Anda.\n"
            "4. Gunakan bahasa profesional, ringkas, dan to-the-point.\n"
            "5. Gunakan Bold (**teks**) dan bullet points untuk keterbacaan.\n"
            "6. DILARANG menggunakan header besar (# atau ##).\n"
            "7. Jangan menambahkan informasi di luar konteks meskipun Anda mengetahuinya.\n\n"
            f"KONTEKS REGULASI POJK 11/2022:\n"
            f"---\n{context_str}\n---"
        )
    }

    # Batasi riwayat chat ke 6 pesan terakhir agar tidak melebihi konteks LLM
    limited_messages = messages[-6:] if len(messages) > 6 else messages
    full_messages    = [system_prompt] + limited_messages

    completion = client.chat.completions.create(
        messages=full_messages,
        model=LLM_MODEL,
        temperature=0.1,
        max_tokens=2048,
        stream=True
    )

    # Filter tag <think> dari model reasoning Qwen secara real-time
    is_thinking  = False
    buffer       = ""

    for chunk in completion:
        content = chunk.choices[0].delta.content
        if content is None:
            continue

        buffer += content

        # Deteksi dan skip blok <think>...</think>
        if "<think>" in buffer:
            is_thinking = True
        if "</think>" in buffer:
            is_thinking  = False
            # Hapus semua konten <think>...</think> dari buffer
            buffer = re.sub(r'<think>.*?</think>', '', buffer, flags=re.DOTALL)

        if not is_thinking and buffer:
            yield buffer
            buffer = ""
            time.sleep(0.01)

    # Flush sisa buffer
    if buffer and not is_thinking:
        clean = re.sub(r'<think>.*?</think>', '', buffer, flags=re.DOTALL).strip()
        if clean:
            yield clean

# ===================================================================
# TAMPILAN ANTARMUKA (USER INTERFACE)
# ===================================================================

# --- CSS Custom ---
st.markdown("""
<style>
    /* Sembunyikan footer Streamlit */
    footer {visibility: hidden;}

    /* Badge sistem */
    .system-badge {
        background: #1a1a2e;
        color: #e0e0e0;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 11px;
        border: 1px solid #333;
        display: inline-block;
        margin: 2px;
    }
    .badge-green { border-color: #2ecc71; color: #2ecc71; }
    .badge-blue  { border-color: #3498db; color: #3498db; }
    .badge-gray  { border-color: #95a5a6; color: #95a5a6; }
</style>
""", unsafe_allow_html=True)

# --- Header ---
st.title("🤖 ARCA")
st.caption("**Audit Regulator Chat Assistant** · POJK No. 11/03/2022")

# --- Sidebar ---
with st.sidebar:
    st.markdown("###🤖 ARCA")
    st.markdown("*Audit Regulator Chat Assistant*")
    st.divider()

    # Template Pertanyaan
    st.subheader("📝 Template Pertanyaan")
    st.write("Gunakan pola berikut untuk hasil optimal:")

    with st.expander("1. Pola Definisi"):
        st.code("Apa aturan tentang [topik]?", language=None)
    with st.expander("2. Validasi Temuan"):
        st.code("Saya menemukan bahwa [masalah]. Apa ketentuan yang relevan?", language=None)
    with st.expander("3. Komparatif"):
        st.code("Apa perbedaan antara [konsep A] dan [konsep B]?", language=None)
    with st.expander("4. Skenario Hipotetis"):
        st.code("Bagaimana jika [skenario]? Apa dampaknya menurut regulasi?", language=None)
    with st.expander("5. Pertanyaan Lanjutan"):
        st.code("Lalu, bagaimana teknis pelaksanaannya?", language=None)

    if st.button("🗑️ Hapus Riwayat Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# --- Area Chat Utama ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Pesan sambutan jika belum ada riwayat
if not st.session_state.messages:
    with st.chat_message("assistant"):
        st.markdown(
            "Halo! Saya **ARCA**, asisten yang siap membantu Anda memahami "
            "**POJK No. 11/POJK.03/2022** tentang Penyelenggaraan Teknologi Informasi "
            "oleh Bank Umum.\n\n"
            "Silakan ajukan pertanyaan Anda seputar regulasi audit TI perbankan. "
            "Gunakan template di sidebar untuk hasil yang lebih optimal. 💬"
        )

# Tampilkan riwayat chat
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "references" in message and message["references"]:
            with st.expander("📚 Lihat Referensi Pasal"):
                st.markdown(message["references"])

# --- Input Chat ---
if prompt := st.chat_input("Tanyakan sesuatu tentang POJK 11..."):

    # Tampilkan pesan user
    st.chat_message("user").markdown(prompt)

    with st.chat_message("assistant"):

        # 1. RETRIEVAL
        with st.spinner("🔍 Menelusuri regulasi..."):
            final_context_df, raw_results = retrieve_context(prompt)

        # 2. GENERATION
        if not final_context_df.empty:
            context_list = final_context_df['teks_konten'].tolist()

            # Siapkan riwayat untuk LLM
            history = [
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ]
            history.append({"role": "user", "content": prompt})

            # Streaming response
            response = st.write_stream(
                generate_answer_stream(history, context_list)
            )

            # Siapkan referensi pasal
            display_df   = final_context_df.drop_duplicates(
                subset=['id_sumber']
            ).head(MAX_CONTEXT)

            source_details = "**Referensi Pasal Terkait:**\n\n"
            for _, row in display_df.iterrows():
                snippet = row['teks_konten'][:400].replace('\n', ' ')
                jarak   = round(row['distance'], 3)
                source_details += (
                    f"📍 **{row['id_sumber']}** "
                    f"> {snippet}...\n\n"
                )

            with st.expander("📚 Lihat Referensi Pasal"):
                st.markdown(source_details)

        else:
            # Pertanyaan di luar cakupan POJK 11
            response = (
                "Maaf, ARCA tidak menemukan rujukan yang relevan dalam "
                "**POJK No. 11/POJK.03/2022** untuk pertanyaan tersebut.\n\n"
                "Kemungkinan pertanyaan Anda berada di luar cakupan regulasi ini. "
                "Silakan coba pertanyaan lain yang berkaitan dengan "
                "penyelenggaraan TI oleh Bank Umum."
            )
            st.markdown(response)
            source_details = None

    # 3. SIMPAN KE RIWAYAT
    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })
    assistant_msg = {
        "role": "assistant",
        "content": response
    }
    if source_details:
        assistant_msg["references"] = source_details
    st.session_state.messages.append(assistant_msg)