import streamlit as st
import sqlite3
import datetime
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from underthesea import word_tokenize

# --- 1. CẤU HÌNH GIAO DIỆN ---
st.set_page_config(
    page_title="Vietnamese Sentiment Assistant",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- 2. CSS TÙY CHỈNH ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&display=swap');
    html, body, [class*="css"] { font-family: 'Roboto', sans-serif; }
    
    .stApp { background-color: #f4f6f9; }
    
    /* Card kết quả: Bỏ icon, chữ to hơn, căn giữa */
    .result-card {
        background-color: white;
        padding: 50px 20px;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        text-align: center;
        border: 2px solid #eee;
        height: 100%;
        display: flex;
        align-items: center;
        justify-content: center;
        flex-direction: column;
    }
    
    .stButton>button {
        background-color: #007bff; color: white; border-radius: 8px; 
        height: 3em; font-weight: bold; border: none; width: 100%;
    }
    .stButton>button:hover { background-color: #0056b3; }
</style>
""", unsafe_allow_html=True)

# --- 3. LOAD MODEL ---
MODEL_NAME = "wonrax/phobert-base-vietnamese-sentiment"

@st.cache_resource
def load_sentiment_pipeline():
    with st.spinner("Đang khởi động hệ thống AI..."):
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
        nlp_pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
        return nlp_pipeline

try:
    classifier = load_sentiment_pipeline()
except Exception as e:
    st.error(f"Lỗi tải model: {e}")
    st.stop()

# --- 4. DATABASE & LOGIC ---
def init_db():
    conn = sqlite3.connect('sentiments.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS sentiments
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, 
                  text TEXT, sentiment TEXT, timestamp TEXT)''')
    conn.commit()
    conn.close()

def save_to_db(text_to_save, sentiment):
    conn = sqlite3.connect('sentiments.db')
    c = conn.cursor()
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute("INSERT INTO sentiments (text, sentiment, timestamp) VALUES (?, ?, ?)",
              (text_to_save, sentiment, now))
    conn.commit()
    conn.close()

def get_history(limit_num):
    conn = sqlite3.connect('sentiments.db')
    c = conn.cursor()
    query = f"SELECT id, text, sentiment, timestamp FROM sentiments ORDER BY id DESC LIMIT {limit_num}"
    c.execute(query)
    data = c.fetchall()
    conn.close()
    return data

init_db()

def preprocess_text(text):
    text = text.lower().strip()
    correction_dict = {
        "hom nay": "hôm nay", "moi nguoi": "mọi người", "rat vui": "rất vui",
        "toi": "tôi", "buon": "buồn", "met moi": "mệt mỏi", "binh thuong": "bình thường",
        "rat": "rất", "qua": "quá", "do": "dở", "khong": "không", "thich": "thích",
        "cam on": "cảm ơn", "xin chao": "xin chào", "mat": "mát", "vai": "vãi",
        "cong viec": "công việc", "on dinh": "ổn định", "ngay mai": "ngày mai",
        "di hoc": "đi học", "that bai": "thất bại"
    }
    for k_dau, co_dau in correction_dict.items():
        text = text.replace(k_dau, co_dau)
    text_tokenized = word_tokenize(text, format="text")
    return text_tokenized

# Hàm map label UI (ĐÃ BỎ ICON)
def map_label_ui(label):
    if label == "POS": return "POSITIVE (Tích cực)", "#2ecc71" # Xanh
    if label == "NEG": return "NEGATIVE (Tiêu cực)", "#e74c3c" # Đỏ
    return "NEUTRAL (Trung tính)", "#f1c40f" # Vàng

# --- 5. GIAO DIỆN CHÍNH ---
st.title("Vietnamese Sentiment Assistant")
st.markdown("Hệ thống phân tích cảm xúc tiếng Việt (PhoBERT + Underthesea)")
st.divider()

col_left, col_right = st.columns([1.5, 1], gap="large")

# TRÁI: NHẬP LIỆU
with col_left:
    st.subheader("Nhập văn bản")
    user_input = st.text_area("Nội dung:", height=150, placeholder="Ví dụ: Hôm nay tôi rất vui...", label_visibility="collapsed")
    if user_input:
        st.caption(f"Độ dài: {len(user_input)} ký tự")
    analyze_btn = st.button("Phân loại")

# Logic
result_package = None
if analyze_btn:
    if not user_input or len(user_input.strip()) < 5:
        st.toast("Câu phải có ít nhất 5 ký tự)!")
    else:
        with st.spinner("Đang xử lý..."):
            processed_text_ai = preprocess_text(user_input)
            result = classifier(processed_text_ai)
            score = result[0]['score']
            label_raw = result[0]['label']
            
            # Logic score < 0.5
            if score < 0.5:
                label_text, color = "NEUTRAL (Trung tính)", "#f1c40f"
            else:
                label_text, color = map_label_ui(label_raw)
            
            # Save DB
            text_for_db = processed_text_ai.replace("_", " ")
            save_to_db(text_for_db, label_text)
            
            result_package = {"text": label_text, "color": color}
            if 'history_limit' not in st.session_state: st.session_state['history_limit'] = 10

# PHẢI: KẾT QUẢ (KHÔNG ICON)
with col_right:
    st.subheader("Kết quả")
    if result_package:
        st.markdown(f"""
        <div class="result-card" style="border-top: 6px solid {result_package['color']};">
            <h2 style="color: {result_package['color']}; margin: 0; font-size: 2.5rem; line-height: 1.2;">
                {result_package['text']}
            </h2>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("Nhập nội dung và bấm Phân loại")

# --- 6. LỊCH SỬ (NÚT XEM THÊM Ở GIỮA) ---
st.divider()
st.subheader("Lịch sử phân loại")

if 'history_limit' not in st.session_state:
    st.session_state['history_limit'] = 10

data = get_history(st.session_state['history_limit'])

if data:
    df = pd.DataFrame(data, columns=["ID", "Câu đã xử lý", "Cảm xúc", "Thời gian"])
    df["ID"] = df["ID"].astype(str)
    
    st.dataframe(
        df[["ID", "Câu đã xử lý", "Cảm xúc"]], 
        hide_index=True, 
        use_container_width=True
    )
    
    st.write("") # Tạo khoảng trống nhỏ
    
    # Kỹ thuật chia cột [5, 2, 5] để nút nằm chính giữa
    c_left, c_center, c_right = st.columns([5, 2, 5])
    
    with c_center:
        if st.button("Xem thêm", use_container_width=True):
            st.session_state['history_limit'] += 10
            st.rerun()
else:
    st.caption("Chưa có dữ liệu.")