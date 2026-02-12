import streamlit as st
import pandas as pd
import faiss
import time
from sentence_transformers import SentenceTransformer
from groq import Groq

# 1. KONFIGURASI HALAMAN
st.set_page_config(page_title="ARCA - Audit Assistant", layout="centered", page_icon="🤖")
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

# 2. FUNGSI LOAD ASSETS (DENGAN CACHING)
@st.cache_resource
def load_arca_engine():
    model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
    index = faiss.read_index("data/pojk_11_v2.index")
    df = pd.read_pickle("data/pojk_11_dataframe_v2.pkl")
    client = Groq(api_key=GROQ_API_KEY)
    return model, index, df, client

with st.spinner("Menginisialisasi ARCA..."):
    model, index, df, client = load_arca_engine()

# 3. LOGIKA RAG

def search_pojk(query, k=15):
    query_vector = model.encode([query])
    D, I = index.search(query_vector, k)
    results_df = df.iloc[I[0]].copy()
    results_df['distance'] = D[0]
    results_df = results_df.drop_duplicates(subset=['teks_konten'])
    return results_df

def generate_answer_stream(messages, context):
    context_str = "\n\n".join(context)
    
    system_prompt = {
        "role": "system",
        "content": f"""Anda adalah ARCA, asisten audit TI. 
        Tugas Anda: Menjawab pertanyaan HANYA berdasarkan KONTEKS yang disediakan.
        
        ATURAN KETAT:
        1. Jangan memberikan langkah-langkah atau saran yang TIDAK tertulis secara eksplisit dalam KONTEKS.
        2. Jika pertanyaan lanjutan tidak relevan dengan KONTEKS, katakan Anda tidak menemukan detailnya.
        3. DILARANG menggunakan Header besar (# atau ##). Gunakan Bold (**teks**) dan Bullet Points.
        4. Jika jawaban panjang, bagi menjadi maksimal 3-4 poin ringkas agar tidak menumpuk.
        5. Selalu konsisten dengan gaya bahasa profesional dan to-the-point.
        
        KONTEKS DOKUMEN:
        ---
        {context_str}
        ---"""
    }
    
    limited_messages = messages[-5:] if len(messages) > 5 else messages
    full_messages = [system_prompt] + limited_messages
    
    completion = client.chat.completions.create(
        messages=full_messages,
        model="qwen/qwen3-32b",
        temperature=0.1, 
        max_tokens=2048,
        stream=True 
    )
    
    is_thinking = False
    for chunk in completion:
        content = chunk.choices[0].delta.content
        if content:
            if "<think>" in content: is_thinking = True
            if not is_thinking:
                yield content
                time.sleep(0.02)
            if "</think>" in content: is_thinking = False

# ===================================================================
# 4. TAMPILAN ANTARMUKA (USER INTERFACE)
# ===================================================================
st.title("🤖 ARCA")
st.caption("Audit Regulator Chat Assistant - POJK No. 11/POJK.03/2022")

# --- SIDEBAR: LOGO, INSTRUKSI, & TEMPLATE ---
with st.sidebar:

    # Menu Template Pertanyaan
    st.subheader("📝 Template Pertanyaan")
    st.write("gunakan pola untuk mendapat hasil yang maksimal:")

    with st.expander("1. Pola Definisi"):
        st.code("Apa aturan tentang [topik]?", language=None)
        
    with st.expander("2. Validasi Temuan"):
        st.code("Saya menemukan bahwa [masalah]. Apa ketentuan yang relevan?", language=None)
        
    with st.expander("3. Komparatif / Pengecualian"):
        st.code("Apa perbedaan antara [konsep A] dan [konsep B]?", language=None)

    with st.expander("4. Skenario Hipotetis"):
        st.code("Bagaimana jika [skenario]? Apa dampaknya menurut regulasi?", language=None)
        
    with st.expander("5. Pertanyaan Lanjutan"):
        st.code("Lalu, bagaimana teknis pelaksanaannya?", language=None)

    st.divider()
    if st.button("🗑️ Hapus Riwayat Chat"):
        st.session_state.messages = []
        st.rerun()

# --- AREA CHAT UTAMA ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Tampilkan riwayat chat
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "references" in message:
            with st.expander("📚 Lihat Referensi"):
                st.markdown(message["references"])

# Input Chat
if prompt := st.chat_input("Tanyakan sesuatu tentang POJK 11..."):
    st.chat_message("user").markdown(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner("ARCA sedang menelusuri dokumen..."):
            # 1. Retrieval
            raw_results = search_pojk(prompt)
            
            highly_relevant = raw_results[raw_results['distance'] <= 6.0]
            if not highly_relevant.empty:
                final_context_df = highly_relevant
            else:
                moderately_relevant = raw_results[raw_results['distance'] <= 7.0]
                final_context_df = moderately_relevant.head(3) if not moderately_relevant.empty else pd.DataFrame()

        # 2. Generation & Streaming
        if not final_context_df.empty:
            display_df = final_context_df.drop_duplicates(subset=['id_sumber']).head(5)
            context_list = final_context_df['teks_konten'].tolist()
            
            history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
            history.append({"role": "user", "content": prompt})
            
            # Efek mengetik
            response = st.write_stream(generate_answer_stream(history, context_list))
            
            # Siapkan Referensi
            source_details = "**Referensi Pasal Terkait:**\n"
            for _, row in display_df.iterrows():
                snippet = row['teks_konten'][:300].replace('\n', ' ')
                source_details += f"\n📍 **{row['id_sumber']}**\n> {snippet}...\n"
            
            with st.expander("📚 Lihat Referensi"):
                st.markdown(source_details)
        else:
            response = "Maaf, ARCA tidak menemukan referensi yang relevan dalam dokumen POJK 11 mengenai pertanyaan tersebut."
            st.markdown(response)
            source_details = None

    # 3. Simpan ke Riwayat
    st.session_state.messages.append({"role": "user", "content": prompt})
    assistant_msg = {"role": "assistant", "content": response}
    if source_details:
        assistant_msg["references"] = source_details
    st.session_state.messages.append(assistant_msg)

