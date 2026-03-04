import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ============================
# --- Page Config ---
# ============================

st.set_page_config(
    page_title="Bank DELTA Term Deposit Prediction",
    page_icon="🏦",
    layout="centered"
)

# ============================
# --- Load Model ---
# ============================

@st.cache_resource
def load_model():
    with open('models/XGBOOST_ncr_tuned_fs.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

model = load_model()

# ============================
# --- Feature Engineering ---
# ============================

def engineer_features(df):
    df = df.copy()

    # campaign_group (dari campaign)
    df['campaign_group'] = pd.cut(
        df['campaign'],
        bins=[0, 2, 4, 7, 11, np.inf],
        labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'],
        right=False
    )

    # risk_score — loan sudah di-drop dari feature selection, jadi tidak dipakai
    df['risk_score'] = (
        df['default'].map({'yes': 1, 'no': 0, 'unknown': 0}).fillna(0) +
        df['housing'].map({'yes': 1, 'no': 0, 'unknown': 0}).fillna(0)
    )

    # was_contacted_before (dari pdays, lalu drop pdays)
    df['was_contacted_before'] = df['pdays'].apply(lambda x: 0 if x == 999 else 1)
    df.drop(columns=['pdays'], inplace=True)

    # age_group (dari age)
    df['age_group'] = pd.cut(
        df['age'],
        bins=[0, 30, 45, 60, 100],
        labels=['young', 'adult', 'middle_age', 'senior'],
        right=False
    )

    # month_num (dari month, lalu drop month)
    df['month_num'] = pd.to_datetime(df['month'], format='%b').dt.month
    

    return df

# ============================
# --- Header ---
# ============================

st.title("🏦 Bank DELTA Term Deposit Prediction")
st.markdown("Prediksi apakah nasabah akan **subscribe** term deposit berdasarkan data kampanye.")
st.divider()

# ============================
# --- 1. Data Demografis Nasabah ---
# ============================

st.subheader("📌 1️⃣ Data Demografis Nasabah")

col1, col2 = st.columns(2)

with col1:
    age       = st.number_input("Age", min_value=18, max_value=100, value=35)
    marital   = st.selectbox("Marital Status", ['married', 'single', 'divorced', 'unknown'])
    education = st.selectbox("Education", [
        'basic.4y', 'basic.6y', 'basic.9y', 'high.school',
        'illiterate', 'professional.course', 'university.degree', 'unknown'
    ])

with col2:
    job     = st.selectbox("Job", [
        'admin.', 'blue-collar', 'entrepreneur', 'housemaid',
        'management', 'retired', 'self-employed', 'services',
        'student', 'technician', 'unemployed', 'unknown'
    ])
    default = st.selectbox("Has Credit in Default?", ['no', 'yes', 'unknown'])
    housing = st.selectbox("Has Housing Loan?",      ['no', 'yes', 'unknown'])

st.divider()

# ============================
# --- 2. Data Terkait Kontak Kampanye ---
# ============================

st.subheader("📌 2️⃣ Data Terkait Kontak Kampanye")

col3, col4 = st.columns(2)

with col3:
    
    month       = st.selectbox("Last Contact Month", [
        'jan','feb','mar','apr','may','jun',
        'jul','aug','sep','oct','nov','dec'
    ])
    day_of_week = st.selectbox("Last Contact Day", ['mon', 'tue', 'wed', 'thu', 'fri'])

with col4:
    campaign = st.number_input("Number of Contacts (this campaign)", min_value=1, value=2)
    contact     = st.selectbox("Contact Type", ['cellular', 'telephone'])

st.divider()

# ============================
# --- 3. Riwayat Kampanye Sebelumnya ---
# ============================

st.subheader("📌 3️⃣ Riwayat Kampanye Sebelumnya")

col5, col6 = st.columns(2)

with col5:
    pdays    = st.number_input("Days Since Last Contact (999 = not contacted)", min_value=0, max_value=999, value=999)
    previous = st.number_input("Number of Contacts (previous campaign)", min_value=0, value=0)

with col6:
    poutcome = st.selectbox("Previous Campaign Outcome", ['failure', 'nonexistent', 'success'])

st.divider()

# ============================
# --- 4. Variabel Ekonomi (Makro) ---
# ============================

st.subheader("📌 4️⃣ Variabel Ekonomi (Makro)")
st.caption("Ini adalah indikator ekonomi saat kampanye berlangsung.")

col7, col8 = st.columns(2)

with col7:
    cons_price_idx = st.number_input("Consumer Price Index",      value=93.994, format="%.3f")
    cons_conf_idx  = st.number_input("Consumer Confidence Index", value=-36.4,  format="%.1f")

with col8:
    euribor3m = st.number_input("Euribor 3 Month Rate", value=4.857, format="%.3f")

# ============================
# --- Predict ---
# ============================

st.divider()

if st.button("🔍 Predict", use_container_width=True, type="primary"):

    input_data = pd.DataFrame([{
        'age':            age,
        'job':            job,
        'marital':        marital,
        'education':      education,
        'default':        default,
        'housing':        housing,
        'contact':        contact,
        'month':          month,
        'day_of_week':    day_of_week,
        'campaign':       campaign,
        'pdays':          pdays,
        'previous':       previous,
        'poutcome':       poutcome,
        'cons.price.idx': cons_price_idx,
        'cons.conf.idx':  cons_conf_idx,
        'euribor3m':      euribor3m,
    }])

    # Apply feature engineering
    input_data = engineer_features(input_data)

    prediction = model.predict(input_data)[0]
    proba = float(model.predict_proba(input_data)[0][1])


    threshold = 0.1  # isi sesuai hasil evaluasi kamu
    prediction = 1 if proba >= threshold else 0

    st.subheader("📈 Hasil Prediksi")

    if prediction == 1:
        st.success(f"✅ **AKAN Subscribe** — Probabilitas: `{proba:.2%}`")
        st.progress(proba)
        st.info("💡 Nasabah ini diprediksi **tertarik** dengan term deposit. Prioritaskan untuk dihubungi.")
    else:
        st.error(f"❌ **TIDAK Subscribe** — Probabilitas: `{proba:.2%}`")
        st.progress(proba)
        st.warning("💡 Nasabah ini diprediksi **tidak tertarik**. Pertimbangkan pendekatan berbeda.")

# ============================
# --- Footer ---
# ============================

st.divider()
st.caption("Model: XGboost + NeighbourhoodCleaningRule | Metric: F1")