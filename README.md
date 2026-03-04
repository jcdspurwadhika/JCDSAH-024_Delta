# JCDSAH-024_Delta


# Prediksi Langganan Deposito Berjangka - Bank Marketing Campaign

Proyek akhir analisis data & machine learning untuk memprediksi apakah seorang nasabah akan berlangganan (**subscribe**) produk **deposito berjangka** berdasarkan data kampanye pemasaran telepon sebuah bank di Portugal (2008–2010).

**Tujuan utama proyek ini adalah**:
- Meningkatkan efisiensi kampanye pemasaran dengan memprediksi nasabah potensial
- Mengurangi biaya operasional call center
- Memaksimalkan **conversion rate** dan **ROI** kampanye

## Dataset

**Bank Marketing UCI**  
Sumber: [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/222/bank+marketing)  
Jumlah baris: ~41.188  
Jumlah kolom: 21  
Target: `y` (yes/no) → apakah nasabah berlangganan deposito  
Imbalance: sangat tinggi (~88.3% no, ~11.7% yes)


## Fitur Utama Proyek

- **Exploratory Data Analysis (EDA)** mendalam
- Feature engineering yang signifikan:
  - Pengelompokan `age`, `campaign`, `month`
  - Pembuatan fitur `was_contacted_before`, `month_num`, dll.
- Penanganan **class imbalance** menggunakan **Neighbourhood Cleaning Rule (NCR)**
- Model terbaik: **XGBOOST** dengan hyperparameter tuning
- Metrik evaluasi utama: **F1-score** (karena imbalance)
- Aplikasi web interaktif menggunakan **Streamlit** untuk demo prediksi

## Hasil Model Terbaik

| Model                     | F1-Score (Test) | Precision | Recall | Catatan                       |
|---------------------------|------------------|-----------|--------|-------------------------------|
| XGBOOST + NCR tuned      | ~51.19%          | 51.53%    | 50.86% | Model final yang digunakan    |


