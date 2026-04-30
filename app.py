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
    page_icon="🤖",
    initial_sidebar_state="expanded"
)

# ===================================================================
# KONFIGURASI SISTEM
# Dataset  : Regulasi Only (pojk_regulasi_only)
# Threshold: 5.5 (highly relevant) / 7.0 (fallback)
# LLM      : qwen/qwen3-32b
# ===================================================================
THRESHOLD_HIGHLY_RELEVANT = 5.5
THRESHOLD_FALLBACK        = 7.0
LLM_MODEL                 = "qwen/qwen3-32b"
TOP_K                     = 10
MAX_CONTEXT               = 3

GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

# ===================================================================
# LOAD ASSETS (DENGAN CACHING)
# ===================================================================
@st.cache_resource
def load_arca_engine():
    """Memuat model embedding, FAISS index, dan dataframe ke memori."""
    model  = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
    index  = faiss.read_index("data/pojk_regulasi_only.index")
    df     = pd.read_pickle("data/pojk_regulasi_only_df.pkl")
    client = Groq(api_key=GROQ_API_KEY)
    return model, index, df, client

with st.spinner("⏳ Menginisialisasi ARCA..."):
    model, index, df, client = load_arca_engine()

# ===================================================================
# FUNGSI RAG
# ===================================================================
def search_pojk(query, k=TOP_K):
    """Melakukan semantic search ke FAISS index."""
    query_vector = model.encode([query])
    D, I         = index.search(query_vector, k)
    results_df   = df.iloc[I[0]].copy()
    results_df['distance'] = D[0]
    return results_df

def retrieve_context(query):
    """
    Pipeline retrieval dengan dua threshold:
    - T5.5: Sangat relevan, ambil semua yang lolos
    - T7.0: Fallback, ambil top-3 jika tidak ada yang lolos T5.5
    - Di luar T7.0: Tolak, pertanyaan di luar cakupan POJK 11
    """
    raw_results = search_pojk(query)

    highly_relevant = raw_results[raw_results['distance'] <= THRESHOLD_HIGHLY_RELEVANT]
    if not highly_relevant.empty:
        return highly_relevant.head(MAX_CONTEXT), raw_results

    fallback = raw_results[raw_results['distance'] <= THRESHOLD_FALLBACK]
    if not fallback.empty:
        return fallback.head(3), raw_results

    return pd.DataFrame(), raw_results

def generate_answer_stream(messages, context_list):
    """
    Mengirim konteks dan riwayat ke LLM via Groq API dengan streaming.
    Menghapus tag <think> dari output model reasoning Qwen secara real-time.
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
            "2. Jika informasi tidak ada dalam konteks, nyatakan secara eksplisit bahwa "
            "informasi tersebut tidak tersedia dalam regulasi yang dirujuk.\n"
            "3. Selalu sebutkan pasal yang menjadi dasar jawaban Anda.\n"
            "4. Gunakan bahasa profesional, ringkas, dan to-the-point.\n"
            "5. Gunakan Bold (**teks**) dan bullet points untuk keterbacaan.\n"
            "6. DILARANG menggunakan header besar (# atau ##).\n"
            "7. Jangan menambahkan informasi di luar konteks meskipun Anda mengetahuinya.\n\n"
            f"KONTEKS REGULASI POJK 11/2022:\n"
            f"---\n{context_str}\n---"
        )
    }

    limited_messages = messages[-6:] if len(messages) > 6 else messages
    full_messages    = [system_prompt] + limited_messages

    completion = client.chat.completions.create(
        messages=full_messages,
        model=LLM_MODEL,
        temperature=0.1,
        max_tokens=2048,
        stream=True
    )

    is_thinking = False
    buffer      = ""

    for chunk in completion:
        content = chunk.choices[0].delta.content
        if content is None:
            continue

        buffer += content

        if "<think>" in buffer:
            is_thinking = True
        if "</think>" in buffer:
            is_thinking = False
            buffer = re.sub(r'<think>.*?</think>', '', buffer, flags=re.DOTALL)

        if not is_thinking and buffer:
            yield buffer
            buffer = ""
            time.sleep(0.01)

    if buffer and not is_thinking:
        clean = re.sub(r'<think>.*?</think>', '', buffer, flags=re.DOTALL).strip()
        if clean:
            yield clean

# ===================================================================
# TAMPILAN ANTARMUKA
# ===================================================================
st.title("🤖 ARCA")
st.caption("**Audit Regulation Chat Assistant** · POJK No. 11/POJK.03/2022")

with st.sidebar:
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

    st.divider()
    if st.button("🗑️ Hapus Riwayat Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []

if not st.session_state.messages:
    with st.chat_message("assistant"):
        st.markdown(
            "Halo! Saya **ARCA**, asisten yang siap membantu Anda memahami "
            "**POJK No. 11/POJK.03/2022** tentang Penyelenggaraan Teknologi Informasi "
            "oleh Bank Umum.\n\n"
            "Silakan ajukan pertanyaan Anda seputar regulasi audit TI perbankan. "
            "Gunakan template di sidebar untuk hasil yang lebih optimal."
        )

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "references" in message and message["references"]:
            with st.expander("📚 Lihat Referensi Pasal"):
                st.markdown(message["references"])

if prompt := st.chat_input("Tanyakan sesuatu tentang POJK 11..."):

    st.chat_message("user").markdown(prompt)

    with st.chat_message("assistant"):

        with st.spinner("🔍 Menelusuri regulasi..."):
            final_context_df, raw_results = retrieve_context(prompt)

        if not final_context_df.empty:
            context_list = final_context_df['teks_konten'].tolist()

            history = [
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ]
            history.append({"role": "user", "content": prompt})

            response = st.write_stream(
                generate_answer_stream(history, context_list)
            )

            display_df     = final_context_df.drop_duplicates(
                subset=['id_sumber']
            ).head(MAX_CONTEXT)
            source_details = "**Referensi Pasal Terkait:**\n\n"
            for _, row in display_df.iterrows():
                snippet = row['teks_konten'][:400].replace('\n', ' ')
                source_details += f"📍 **{row['id_sumber']}**\n> {snippet}...\n\n"

            with st.expander("📚 Lihat Referensi Pasal"):
                st.markdown(source_details)

        else:
            response = (
                "Maaf, ARCA tidak menemukan rujukan yang relevan dalam "
                "**POJK No. 11/POJK.03/2022** untuk pertanyaan tersebut.\n\n"
                "Kemungkinan pertanyaan Anda berada di luar cakupan regulasi ini. "
                "Silakan coba pertanyaan lain yang berkaitan dengan "
                "penyelenggaraan TI oleh Bank Umum."
            )
            st.markdown(response)
            source_details = None

    st.session_state.messages.append({"role": "user", "content": prompt})
    assistant_msg = {"role": "assistant", "content": response}
    if source_details:
        assistant_msg["references"] = source_details
    st.session_state.messages.append(assistant_msg)