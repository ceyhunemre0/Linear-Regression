# app.py

import streamlit as st
import joblib
import numpy as np

# Başlık
st.title("💼 Maaş Tahmini Uygulaması")
st.write("Lütfen tecrübenizi yıl cinsinden girin:")

# Kullanıcıdan giriş al
experience = st.number_input("Tecrübe (yıl):", min_value=0.0, max_value=50.0, value=1.0, step=0.5)

# Butona basıldığında tahmin yapılacak
if st.button("HESAPLA"):
    # Modeli yükle
    model = joblib.load('linear_model.pkl')

    # Tahmin yap
    prediction = model.predict(np.array([[experience]]))

    # Sonucu göster
    st.success(f"✅ Tahmini Maaş: {prediction[0]:,.2f} ₺")
