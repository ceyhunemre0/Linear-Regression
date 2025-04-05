# train_model.py

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import joblib  # Modeli kaydetmek için kullanılır

# 1. Veriyi oku
df = pd.read_csv('Salary_dataset.csv')

# 2. Değişkenleri ayır
X = df[['YearsExperience']]  # 2D
y = df['Salary']             # 1D

# 3. Eğitim ve test seti
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Modeli oluştur ve eğit
model = LinearRegression()
model.fit(X_train, y_train)

# 5. Modeli kaydet
joblib.dump(model, 'linear_model.pkl')

print("Model başarıyla eğitildi ve 'linear_model.pkl' olarak kaydedildi.")
