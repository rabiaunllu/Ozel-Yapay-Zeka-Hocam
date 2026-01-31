import streamlit as st
import google.generativeai as genai
import os
import time
import tempfile
import re
from fpdf import FPDF

# --- 1. AYARLAR VE KÜTÜPHANELER ---
st.set_page_config(
    page_title="Yapay Zeka Hocam",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. GÖRSEL TASARIM (REACT & TAILWIND TARZI CSS) ---
st.markdown(
    """
    <style>
    /* Genel Ayarlar & Fontlar */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    
    .stApp {
        background-color: #0e1117;
        color: #f3f4f6;
        font-family: 'Inter', sans-serif;
    }
    
    /* Standart Streamlit Başlıklarını Gizle */
    footer, #MainMenu {visibility: hidden;}
    .st-emotion-cache-16txtl3 {padding-top: 0rem !important;} /* Üst boşluğu al */
    
    /* ÖZEL BAŞLIK ALANI */
    .custom-header {
        padding: 1.5rem;
        border-bottom: 1px solid #1f2937;
        background-color: #0e1117;
        margin-bottom: 1rem;
    }
    
    /* SEKMELER (TABS) - React Tarzı */
    button[data-baseweb="tab"] {
        font-size: 16px !important;
        font-weight: 600 !important;
        color: #9ca3af !important; /* gray-400 */
        background-color: transparent !important;
        border: none !important;
        border-bottom: 2px solid transparent !important;
        border-radius: 0 !important;
        padding: 10px 24px !important;
    }
    
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #60a5fa !important; /* blue-400 */
        border-bottom: 2px solid #60a5fa !important;
        background-color: rgba(31, 41, 55, 0.5) !important; /* Hafif koyu arka plan */
    }
    
    /* SOHBET BALONLARI */
    .chat-container {
        display: flex;
        flex-direction: column;
        gap: 12px;
        padding-bottom: 20px;
    }
    
    /* Kullanıcı Mesajı: Mavi, Sağa Yaslı, Yuvarlak */
    .chat-bubble-user {
        background-color: #2563EB; /* Tailwind Blue-600 */
        color: white;
        padding: 12px 18px;
        border-radius: 18px;
        border-bottom-right-radius: 4px;
        max-width: 75%;
        align-self: flex-end;
        margin-left: auto;
        font-size: 15px;
        line-height: 1.5;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Hoca Mesajı: Koyu Gri, Sola Yaslı */
    .chat-bubble-model {
        background-color: #1F2937; /* Tailwind Gray-800 */
        color: #F3F4F6; /* Gray-100 */
        padding: 12px 18px;
        border-radius: 18px;
        border-bottom-left-radius: 4px;
        max-width: 75%;
        align-self: flex-start;
        margin-right: auto;
        font-size: 15px;
        line-height: 1.5;
        border: 1px solid #374151; /* Gray-700 border */
    }
    
    /* Sidebar Butonları */
    .stButton > button {
        width: 100%;
        border-radius: 8px;
        font-weight: 600;
    }
    
    /* Reset Butonuna Özel Stil (Kırmızımsı Hover) - CSS Hack */
    section[data-testid="stSidebar"] .stButton > button:hover {
        border-color: #ef4444 !important;
        color: #ef4444 !important;
        background-color: rgba(239, 68, 68, 0.1) !important;
    }
    
    </style>
    """,
    unsafe_allow_html=True
)

# --- 3. FONKSİYONLAR ---

# --- PRIORITY LIST (UPDATED: Pro REMOVED) ---
PRIORITY_MODELS = [
    "gemini-2.5-flash",    # Try newest first
    "gemini-2.0-flash",    # Validated as available
    "gemini-1.5-flash"     # Standard fallback
    # gemini-pro REMOVED to prevent 404/Fallback error
]

def clean_text_for_pdf(text):
    """
    Cleans raw Markdown/LaTeX artifacts and handles Turkish char replacement for PDF.
    """
    # 1. Regex Cleanup
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)  # **bold**
    text = re.sub(r'\*(.*?)\*', r'\1', text)      # *italic*
    text = re.sub(r'__(.*?)__', r'\1', text)      # __bold__
    text = re.sub(r'^#+\s*', '', text, flags=re.MULTILINE) # Headers
    text = re.sub(r'\$|\\\[|\\\]|\\\(|\\\)', '', text) # LaTeX
    
    # 2. Turkish Character Replacement
    text = text.replace("ğ","g").replace("Ğ","G").replace("ş","s").replace("Ş","S")
    text = text.replace("ı","i").replace("İ","I").replace("ö","o").replace("Ö","O")
    text = text.replace("ü","u").replace("Ü","U").replace("ç","c").replace("Ç","C")
    
    # 3. Encoding Safe
    return text.encode('latin-1', 'replace').decode('latin-1')

def upload_files_helper(uploaded_files):
    """Helper to upload files to Gemini and return file parts"""
    file_parts = []
    for pdf in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(pdf.getvalue())
            tmp_path = tmp.name
        
        g_file = genai.upload_file(tmp_path, mime_type="application/pdf")
        file_parts.append(g_file)
    return file_parts

def connect_to_best_model(api_key, uploaded_files, system_instruction):
    """
    WATERFALL STRATEGY: Iterates through PRIORITY_MODELS to find a working model.
    Returns: (connected_model_name, chat_session, file_parts)
    """
    genai.configure(api_key=api_key)
    
    # 1. Upload Files
    try:
        file_parts = upload_files_helper(uploaded_files)
    except Exception as e:
        st.error(f"Dosya yükleme hatası: {e}")
        return None, None, None

    # 2. Waterfall Connection Loop
    for model_name in PRIORITY_MODELS:
        try:
            model = genai.GenerativeModel(model_name, system_instruction=system_instruction)
            
            # Start Chat Session
            chat = model.start_chat(
                history=[
                    {
                        "role": "user",
                        "parts": file_parts + ["Bu dökümanları analiz et ve bekle."]
                    },
                    {
                        "role": "model",
                        "parts": ["Tamamdır, dökümanlar alındı. Hazırım."]
                    }
                ]
            )
            
            return model_name, chat, file_parts
            
        except Exception:
            # If 429 or 404, we continue to the next model in the list
            continue 
            
    # If all fail
    return None, None, None

def create_pdf(chat_history):
    """Sohbet geçmişini PDF'e çevirir"""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, txt="Ders Notlari ve Sohbet Gecmisi", ln=True, align='C')
    pdf.set_line_width(0.5)
    pdf.line(10, 25, 200, 25)
    pdf.ln(10)
    
    for msg in chat_history:
        if hasattr(msg.parts[0], 'text'):
            role = "HOCA" if msg.role == "model" else "OGRENCI"
            clean_text = clean_text_for_pdf(msg.parts[0].text)
            
            pdf.set_font("Arial", 'B', 10)
            if role == "OGRENCI":
                pdf.set_text_color(0, 0, 150)
            else:
                pdf.set_text_color(50, 50, 50)
            pdf.cell(0, 8, f"{role}:", ln=True)
            
            pdf.set_font("Arial", '', 10)
            pdf.set_text_color(0, 0, 0)
            pdf.multi_cell(0, 6, clean_text)
            pdf.ln(5)
            
    return pdf.output(dest='S').encode('latin-1')

def create_quiz_pdf(quiz_text):
    """Sınav sonucunu PDF'e çevirir"""
    pdf = FPDF()
    pdf.add_page()
    
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, txt="Yapay Zeka Hocam - Sinav Kagidi", ln=True, align='C')
    pdf.set_line_width(0.5)
    pdf.line(10, 25, 200, 25)
    pdf.ln(10)
    
    clean_text = clean_text_for_pdf(quiz_text)
    
    pdf.set_font("Arial", '', 11)
    pdf.set_text_color(0, 0, 0)
    pdf.multi_cell(0, 6, clean_text)
            
    return pdf.output(dest='S').encode('latin-1')

# --- 4. ANA UI & LOGIC ---

def main():
    # Session States
    if "chat_session" not in st.session_state:
        st.session_state.chat_session = None
    if "config_hash" not in st.session_state:
        st.session_state.config_hash = "" # Tracks Files + Prompt changes
    if "file_cache" not in st.session_state:
        st.session_state.file_cache = None
    if "active_model" not in st.session_state:
        st.session_state.active_model = None
    if "last_quiz" not in st.session_state:
        st.session_state.last_quiz = None

    # --- SIDEBAR ---
    with st.sidebar:
        st.header("⚙️ Ayarlar")
        
        st.subheader("🎭 Hoca Kişiliği")
        secenekler = [
            'Arkadaş Canlısı & Basit Anlatan', 
            'Disiplinli & Akademik Profesör', 
            'Sokratik (Sürekli Soru Soran)', 
            'Sınav Hazırlayıcı (Zor)', 
            '✍️ Kendi Tarifim (Özel Prompt)'
        ]
        hoca_tarzi = st.selectbox("Seçimini Yap:", secenekler, label_visibility="collapsed")
        
        system_instruction = ""
        if hoca_tarzi == '✍️ Kendi Tarifim (Özel Prompt)':
            system_instruction = st.text_area("Hocayı Tarif Et:", placeholder="Örn: Çok detaycı...")
        else:
            prompts = {
                'Arkadaş Canlısı & Basit Anlatan': "Sen samimi, yardımsever ve basit anlatan bir öğretmensin.",
                'Disiplinli & Akademik Profesör': "Sen ciddi, akademik ve resmi bir profesörsün.",
                'Sokratik (Sürekli Soru Soran)': "Asla cevabı direkt verme. Sorular sorarak öğrenciyi yönlendir.",
                'Sınav Hazırlayıcı (Zor)': "Öğrenciyi zorla. Detaylardan şaşırtmacalı sorular sor."
            }
            system_instruction = prompts.get(hoca_tarzi, "Yardımcı asistan ol.")
            
        st.divider()
        st.subheader("📂 Dosya Yükle")
        uploaded_files = st.file_uploader("PDF Seç", type=['pdf'], accept_multiple_files=True, label_visibility="collapsed")
        
        api_key = st.secrets.get("GOOGLE_API_KEY")
        if not api_key:
            st.warning("API Key Eksik!")
            api_key = st.text_input("Google API Key:", type="password")
            
        st.write("") 
        st.write("") 
        
        if "reset_confirm" not in st.session_state:
            st.session_state.reset_confirm = False

        if st.button("🗑️ Sohbeti Sıfırla"):
            st.session_state.reset_confirm = True
            st.rerun()
            
        if st.session_state.reset_confirm:
            st.warning("Bu işlem geri alınamaz! Emin misin?")
            col_yes, col_no = st.columns(2)
            with col_yes:
                if st.button("🔴 Evet, Hepsini Sil"):
                    st.session_state.chat_session = None
                    st.session_state.last_quiz = None
                    st.session_state.file_cache = None
                    st.session_state.active_model = None
                    st.session_state.config_hash = ""
                    st.session_state.reset_confirm = False
                    st.rerun()
            with col_no:
                if st.button("⚪ Vazgeç"):
                    st.session_state.reset_confirm = False
                    st.rerun()

    # --- MODEL CONNECTION LOGIC (PERSISTENT & WATERFALL) ---
    if api_key and uploaded_files:
        # Create a signature for current configuration checks
        current_file_sig = ",".join([f.name for f in uploaded_files])
        current_prompt_sig = hoca_tarzi + str(len(system_instruction)) # Simple prompt hash
        current_config_hash = current_file_sig + "|" + current_prompt_sig
        
        # Only act if configuration changed (or app just started)
        if st.session_state.config_hash != current_config_hash:
            with st.spinner("Hoca hazırlanıyor..."):
                try:
                    genai.configure(api_key=api_key)
                    
                    # STRICT PERSISTENCE CHECK
                    if st.session_state.active_model:
                        # ACTIVE MODEL EXISTS -> FAST PATH
                        # LOCK: We DO NOT search for a model. We use the stored one.
                        model_name = st.session_state.active_model
                        
                        # Determine if we need to re-upload files
                        if not st.session_state.config_hash.startswith(current_file_sig + "|"):
                             files = upload_files_helper(uploaded_files) # Files changed
                        else:
                             # Files are same, reuse cache if valid
                             if st.session_state.file_cache:
                                 files = st.session_state.file_cache
                             else:
                                 files = upload_files_helper(uploaded_files)
                        
                        # Re-initialize Session with NEW Prompt but SAME Model
                        # Note: We removed gemini-pro from priority list, so we assume flash here.
                        model = genai.GenerativeModel(model_name, system_instruction=system_instruction)
                            
                        chat = model.start_chat(
                            history=[
                                {"role": "user", "parts": files + ["Bu dökümanları analiz et ve bekle."]},
                                {"role": "model", "parts": ["Tamamdır, dökümanlar alındı. Hazırım."]}
                            ]
                        )
                    
                    else:
                        # NO ACTIVE MODEL -> WATERFALL PATH (First Run Only)
                        model_name, chat, files = connect_to_best_model(api_key, uploaded_files, system_instruction)
                        
                        if not model_name:
                            st.error("Hiçbir model bağlanamadı. (2.5, 2.0 veya 1.5 Flash yanıt vermedi).")
                            st.stop()
                    
                    # Store State
                    st.session_state.chat_session = chat
                    st.session_state.active_model = model_name
                    st.session_state.file_cache = files
                    st.session_state.config_hash = current_config_hash
                    
                    st.toast(f"Bağlandı: {model_name}", icon="🟢")
                    
                except Exception as e:
                    st.error(f"Beklenmeyen bir hata oluştu: {e}")

    # --- ANA EKRAN YAPISI ---
    
    # 1. Header
    st.markdown(
        """
        <div class="custom-header">
            <h1 style="font-size: 2rem; font-weight: 700; color: white; margin: 0;">🎓 Yapay Zeka Hocam</h1>
            <p style="color: #9ca3af; margin-top: 5px; font-size: 1rem;">Notlarını yükle, sınava hazırlan.</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # 2. Tabs
    tab_chat, tab_exam, tab_about = st.tabs(["💬 Sohbet Odası", "📝 Sınav Modu", "ℹ️ Hakkında"])
    
    # --- TAB 1: SOHBET ODASI ---
    with tab_chat:
        col_title, col_dl = st.columns([8, 2])
        with col_title:
             pass 
        with col_dl:
            if st.session_state.chat_session and len(st.session_state.chat_session.history) > 2:
                pdf_data = create_pdf(st.session_state.chat_session.history)
                st.download_button(
                    label="📥 PDF İndir",
                    data=pdf_data,
                    file_name="sohbet_gecmisi.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
        
        chat_placeholder = st.container()
        
        with chat_placeholder:
            if st.session_state.chat_session:
                messages_html = '<div class="chat-container">'
                
                for msg in st.session_state.chat_session.history:
                    if hasattr(msg.parts[0], 'text'):
                        role = msg.role
                        txt = msg.parts[0].text
                        
                        if "Bu dökümanları analiz et" in txt or "Tamamdır, dökümanlar alındı" in txt:
                            continue

                        if role == "user":
                            messages_html += f'<div class="chat-bubble-user">{txt}</div>'
                        else:
                            messages_html += f'<div class="chat-bubble-model">{txt}</div>'
                
                messages_html += '</div>'
                st.markdown(messages_html, unsafe_allow_html=True)
            else:
                 if not uploaded_files:
                     st.markdown(
                         """
                         <div style="text-align: center; color: #6b7280; margin-top: 50px;">
                            <h3>👈 Önce sol taraftan PDF yükle.</h3>
                         </div>
                         """, unsafe_allow_html=True
                     )

        st.markdown("---")
        user_input = st.chat_input("Hocana bir soru sor...")
        
        if user_input and st.session_state.chat_session:
            st.markdown(f'<div class="chat-bubble-user" style="margin-left:auto;">{user_input}</div>', unsafe_allow_html=True)
            
            try:
                with st.status("Hoca cevaplıyor...", expanded=True) as status:
                    st.write("📂 Dokümanlar taranıyor...")
                    time.sleep(0.5)
                    st.write("🧠 Cevap kurgulanıyor...")
                    
                    response = st.session_state.chat_session.send_message(user_input)
                    
                    status.update(label="Yanıtlandı! ✨", state="complete", expanded=False)
                
                st.rerun()
            except Exception as e:
                st.error(f"Cevap alınamadı: {e}")

    # --- TAB 2: SINAV MODU (İZOLASYONLU) ---
    with tab_exam:
        st.markdown('<div style="background-color: #111827; padding: 20px; border-radius: 10px; border: 1px solid #374151;">', unsafe_allow_html=True)
        st.subheader("📝 Sınav Oluşturucu")
        
        if st.session_state.active_model and st.session_state.file_cache:
            
            c1, c2, c3 = st.columns(3)
            with c1:
                q_count = st.slider("Soru Sayısı", 3, 20, 5)
            with c2:
                difficulty = st.selectbox("Zorluk Seviyesi", ["Kolay", "Orta", "Zor", "Acımasız (Akademik)"])
            with c3:
                q_type = st.selectbox("Soru Tipi", ["Test (Çoktan Seçmeli)", "Klasik (Yazılı)", "Karma (Karışık)"])
            
            st.write("")
            custom_instruction = st.text_area("Sınav İçeriği İçin Özel İstek (Opsiyonel)", placeholder="Örn: Sorular yoruma dayalı, çıkmış soru benzeri olsun...")
            
            st.write("")
            if st.button("🚀 Sınavı Başlat", use_container_width=True):
                prompt_base = f"Bana yüklenen ders notlarından {q_count} adet {difficulty} seviyesinde soru hazırla."
                
                if custom_instruction:
                    prompt_base += f" ÖZEL İSTEK: {custom_instruction}."
                
                if "Test" in q_type:
                    prompt = f"{prompt_base} Sorular çoktan seçmeli (A,B,C,D,E) şıklı olsun. Her sorunun şıklarını alt alta yaz."
                elif "Klasik" in q_type:
                    prompt = f"{prompt_base} Sorular şıkkı olmayan, yorum yapmayı veya işlem yapmayı gerektiren klasik tipte olsun."
                else: 
                    prompt = f"{prompt_base} Soruların yarısı çoktan seçmeli test, diğer yarısı klasik yazılı sorusu olsun. Karışık bir sınav hazırla."
                
                prompt += " En sona mutlaka ayrı bir başlık altında CEVAP ANAHTARINI ekle."

                try: 
                    with st.status("Sınav hazırlanıyor...", expanded=True) as status:
                        st.write("📂 Dosyalar alınıyor...")
                        
                        temp_model_name = st.session_state.active_model
                        temp_files = st.session_state.file_cache
                        
                        # Removed gemini-pro check since it's removed from priority list anyway
                        temp_gen_model = genai.GenerativeModel(temp_model_name, system_instruction="Sen bir sınav hazırlayıcısın.")
                        
                        temp_chat = temp_gen_model.start_chat(
                            history=[
                                {"role": "user", "parts": temp_files + ["Sınav hazırlamaya hazır ol."]},
                                {"role": "model", "parts": ["Anlaşıldı."]}
                            ]
                        )
                        
                        st.write("🧠 Yapay zeka soruları kurguluyor...")
                        
                        res = temp_chat.send_message(prompt)
                        st.session_state.last_quiz = res.text 
                        
                        status.update(label="Sınav Başarıyla Hazırlandı! 🚀", state="complete", expanded=False)
                    
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Sınav oluşturulamadı: {e}")

            if st.session_state.last_quiz:
                st.markdown("---")
                res_col1, res_col2 = st.columns([3, 1])
                
                with res_col1:
                    st.subheader("📝 İşte Sınavın:")
                
                with res_col2:
                    pdf_bytes = create_quiz_pdf(st.session_state.last_quiz)
                    st.download_button(
                        label="📄 PDF İndir",
                        data=pdf_bytes,
                        file_name="Sinav_Kagidi.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )
                
                st.markdown(st.session_state.last_quiz)
                
        else:
            st.info("Lütfen önce PDF yükleyin.")
            
        st.markdown('</div>', unsafe_allow_html=True)

    # --- TAB 3: HAKKINDA ---
    with tab_about:
        st.markdown("""
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin-bottom: 40px;">
            <div style="background-color: #1F2937; padding: 20px; border-radius: 10px; border: 1px solid #374151; text-align: center;">
                <div style="font-size: 40px; margin-bottom: 10px;">🧠</div>
                <h3 style="color: white; margin: 0 0 10px 0;">Akıllı Analiz</h3>
                <p style="color: #9ca3af; margin: 0; font-size: 0.95rem;">PDF notlarını yükle, anında özet ve cevap al.</p>
            </div>
            <div style="background-color: #1F2937; padding: 20px; border-radius: 10px; border: 1px solid #374151; text-align: center;">
                <div style="font-size: 40px; margin-bottom: 10px;">🎭</div>
                <h3 style="color: white; margin: 0 0 10px 0;">Rol Yapma</h3>
                <p style="color: #9ca3af; margin: 0; font-size: 0.95rem;">İster kanka, ister profesör modunda çalış.</p>
            </div>
            <div style="background-color: #1F2937; padding: 20px; border-radius: 10px; border: 1px solid #374151; text-align: center;">
                <div style="font-size: 40px; margin-bottom: 10px;">📝</div>
                <h3 style="color: white; margin: 0 0 10px 0;">Sınav Hazırlığı</h3>
                <p style="color: #9ca3af; margin: 0; font-size: 0.95rem;">Kendi seviyene uygun testler oluştur.</p>
            </div>
        </div>
        
        <div style="background-color: #111827; padding: 25px; border-radius: 10px; border: 1px solid #374151; margin-bottom: 30px;">
            <h3 style="color: white; border-bottom: 1px solid #374151; padding-bottom: 10px; margin-bottom: 15px;">📚 Nasıl Kullanılır?</h3>
            <ul style="color: #9ca3af; line-height: 1.8; list-style-type: none; padding: 0;">
                <li style="margin-bottom: 8px;">📄 <strong>Ders notunu (PDF) yükle.</strong></li>
                <li style="margin-bottom: 8px;">🎭 <strong>Hoca tipini seç</strong> (Sert hoca, kanka hoca vb.).</li>
                <li style="margin-bottom: 8px;">💬 <strong>Sohbet sekmesinden</strong> sorularını sor.</li>
                <li style="margin-bottom: 8px;">📝 <strong>Sınav sekmesinden</strong> kendini dene!</li>
            </ul>
        </div>
        
        <div style="margin-top: 50px; text-align: center; color: #6b7280; border-top: 1px solid #374151; padding-top: 20px;">
            <p>Geliştirici: <strong>Rabiya</strong></p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()