import streamlit as st
import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt

model = joblib.load("kredi_model.pkl")
feature_isimleri = joblib.load("feature_isimleri.pkl")

@st.cache_resource
def get_explainer():
    return shap.TreeExplainer(model)

explainer = get_explainer()

st.set_page_config(page_title="Kredi Risk Skoru", page_icon="📊", layout="wide")

# ---- SIDEBAR: Girdiler ----
st.sidebar.markdown("#### Yaş")
age = st.sidebar.number_input("Yas", 18, 100, value=18, label_visibility="collapsed")

st.sidebar.markdown("#### Aylık Net Gelir (TL)")
income = st.sidebar.number_input("Gelir", 0, value=0, step=1000, label_visibility="collapsed")

st.sidebar.markdown("#### Kredi Kartı Limit Kullanım Oranı (%)")
revol_yuzde = st.sidebar.slider("Kullanim", 0, 100, 30, label_visibility="collapsed")
revol = revol_yuzde / 100

st.sidebar.markdown("#### Aylık gelirin yüzde kaçı borç/kredi ödemesine gidiyor (%)")
debt_yuzde = st.sidebar.slider("Borc", 0, 100, 35, label_visibility="collapsed")
debt = debt_yuzde / 100

st.sidebar.markdown("#### Açık Kredi / Kart Sayısı")
open_credit = st.sidebar.number_input("AcikKredi", 0, value=0, label_visibility="collapsed")

st.sidebar.markdown("#### Gayrimenkul Kredisi Sayısı")
realestate = st.sidebar.number_input("Gayrimenkul", 0, value=0, label_visibility="collapsed")

st.sidebar.markdown("#### Bakmakla Yükümlü Kişi Sayısı")
dependents = st.sidebar.number_input("Bagimli", 0, value=0, label_visibility="collapsed")

st.sidebar.markdown("#### Geçmiş Ödemelerde Gecikme Sayısı")
toplam_gecikme = st.sidebar.number_input("Gecikme", 0, value=0, label_visibility="collapsed")

hesapla = st.sidebar.button("HESAPLA", type="primary", use_container_width=True)

# ---- ANA EKRAN: Başlık ----
st.title("📊 Kredi Risk Değerlendirme Aracı")
st.markdown("Soldaki bilgileri girin, **kredi risk skorunu** ve bu skoru etkileyen **faktörleri** görün.")
st.markdown("---")

if hesapla:
    gecikme_var = 1 if toplam_gecikme > 0 else 0
    veri = pd.DataFrame([[revol, age, debt, income, open_credit,
                          realestate, dependents, toplam_gecikme, gecikme_var]],
                        columns=feature_isimleri)
    olasilik = model.predict_proba(veri)[0][1]
    yuzde = olasilik * 100

    if yuzde < 15:
        seviye, renk, yorum = "DÜŞÜK RİSK", "green", "Ek değerlendirme gerekmez."
    elif yuzde < 40:
        seviye, renk, yorum = "ORTA RİSK", "orange", "Ödeme alışkanlığı ve profil açısından ek değerlendirme önerilir."
    else:
        seviye, renk, yorum = "YÜKSEK RİSK", "red", "Ödeme alışkanlığı ve profil açısından detaylı değerlendirme gerekir."

    sekme1, sekme2, sekme3 = st.tabs(["📊 Sonuç", "❓ Yardım", "🔧 Teknik"])

    # ---- SEKME 1: Ana Sonuç ----
    with sekme1:
        st.markdown(
            f"<h1 style='color:{renk}; font-size:56px; margin-bottom:0;'>{seviye} <span style='font-size:28px; color:gray;'>(%{yuzde:.1f})</span></h1>",
            unsafe_allow_html=True)
        if renk == "green":
            st.success(yorum)
        elif renk == "orange":
            st.warning(yorum)
        else:
            st.error(yorum)

        st.markdown("---")
        if renk == "green":
            st.markdown("#### ✅ Güçlü Profil")
            o1, o2, o3 = st.columns(3)
            o1.success("💳 Kart kullanımı kontrollü")
            o2.success("📅 Ödeme geçmişi temiz")
            o3.success("👍 Mevcut alışkanlıkları sürdürün")
        elif renk == "orange":
            st.markdown("#### ⚠️ Sınırda Profil")
            o1, o2, o3 = st.columns(3)
            o1.warning("💳 Kart kullanımını düşürün")
            o2.warning("⚖️ Borç/gelir dengesini koruyun")
            o3.warning("📅 Zamanında ödeme yapın")
        else:
            st.markdown("#### 🚨 Riskli Profil")
            o1, o2, o3 = st.columns(3)
            o1.error("📉 Borç ve kart kullanımını azaltın")
            o2.error("📅 Düzenli ödeme telafi eder")
            o3.error("️⚖️ Borç/gelir dengesini koruyun")

    # ---- SEKME 2: Detaylı Analiz (kişiye özel, sade) ----
    with sekme2:
        st.markdown("### Bu Araç Nasıl Çalışır?")
        st.markdown("""
        Bu araç, girilen bilgilere bakarak bir kredinin geri ödenmeme olasılığını tahmin eder. 
        Tahmin, geçmiş gerçek kredi verileriyle eğitilmiş bir yapay zeka modeline dayanır.
        """)

        st.markdown("---")
        st.markdown("### Hangi Bilgiler, Neden Soruluyor?")

        st.markdown("""
        **💳 Kredi Kartı Kullanım Oranı**  
        Limitinizin ne kadarını kullandığınız. Limiti sürekli dolu olan biri, nakit sıkışıklığı yaşıyor olabilir; bu riski artırır.

        **📅 Geçmiş Gecikme Sayısı**  
        Daha önce ödemelerinizi kaç kez geciktirdiğiniz. Geçmiş ödeme davranışı, gelecekteki davranışın en güçlü göstergesidir.

        **⚖️ Borç / Gelir Oranı**  
        Gelirinizin ne kadarının borç ödemesine gittiği. Bu oran yükseldikçe ödeme zorlanma riski artar.

        **👤 Yaş**  
        Daha genç başvuranların kredi geçmişi genelde daha kısadır; bu da belirsizliği artırabilir.

        **💰 Gelir**  
        Aylık geliriniz. Daha yüksek ve istikrarlı gelir, geri ödeme gücünü destekler.

        **🏦 Açık Kredi / Gayrimenkul Sayısı**  
        Mevcut kredi ve kart sayınız. Çok sayıda açık kredi, toplam yükü artırabilir.

        **👨‍👩‍👧 Bakmakla Yükümlü Kişi Sayısı**  
        Bakmakla yükümlü olduğunuz kişi sayısı, harcama yükünü etkileyen bir faktördür.
        """)

        st.markdown("---")
        st.info("ℹ️ Bu araç bir ön değerlendirme aracıdır; kesin kredi kararı yerine geçmez. "
                "Sonuç, girilen bilgilere göre bir risk tahmini sunar.")

    # ---- SEKME 3: Teknik ----
    with sekme3:
        st.markdown("### Teknik Bilgi")
        st.markdown("""
        - **Model:** XGBoost (GridSearch ile optimize edildi)
        - **Test ROC-AUC:** ~0.87
        - **Açıklanabilirlik:** SHAP (TreeExplainer)
        - **Veri:** Give Me Some Credit (gerçek veri, TR gelir bağlamına ölçeklendi)
        - **Sınıf dengesizliği:** scale_pos_weight ile ele alındı
        - - **Detaylı metodoloji:** [GitHub README](https://github.com/meteoguz13/Credit-Risk-Analysis#readme)
        """)

else:
    st.markdown("""
    <div style='text-align:center; padding:40px; background-color:#1a1a2e; border-radius:15px; margin-top:20px;'>
        <h2>👈 Başlamak için soldaki bilgileri girin</h2>
        <p style='font-size:18px; color:gray;'>Değerleri girip <b>HESAPLA</b> butonuna basın, sonucu anında görün.</p>
    </div>
    """, unsafe_allow_html=True)