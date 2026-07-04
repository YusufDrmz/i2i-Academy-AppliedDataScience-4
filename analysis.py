import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────
# 1. VERİYİ YÜKLE
# ─────────────────────────────────────────────
train_df = pd.read_csv("kdd_train.csv")
test_df  = pd.read_csv("kdd_test.csv")

print("Train seti:", train_df.shape)
print("Test seti :", test_df.shape)

# ─────────────────────────────────────────────
# 2. KATEGORİK SÜTUNLARI ENCODE ET
#    (protocol_type, service, flag → sayıya çevir)
# ─────────────────────────────────────────────
categorical_cols = ["protocol_type", "service", "flag"]

le = LabelEncoder()
for col in categorical_cols:
    # Train ve test'i birlikte fit ediyoruz ki aynı encoding kullansınlar
    combined = pd.concat([train_df[col], test_df[col]])
    le.fit(combined)
    train_df[col] = le.transform(train_df[col])
    test_df[col]  = le.transform(test_df[col])

# ─────────────────────────────────────────────
# 3. FEATURE VE TARGET'I AYIR
# ─────────────────────────────────────────────
X_train = train_df.drop(columns=["labels"])
y_train = train_df["labels"]

X_test  = test_df.drop(columns=["labels"])
y_test  = test_df["labels"]

# ─────────────────────────────────────────────
# 4. FEATURE SCALING
#    KNN için zorunlu, RF için de uyguluyoruz (tutarlılık)
#    fit_transform sadece train'de, test'te sadece transform!
# ─────────────────────────────────────────────
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# ─────────────────────────────────────────────
# 5. MODELLERİ EĞİT
# ─────────────────────────────────────────────
print("\nModeller eğitiliyor...")

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)
print("KNN eğitimi tamamlandı.")

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_scaled, y_train)
print("Random Forest eğitimi tamamlandı.")

# ─────────────────────────────────────────────
# 6. TAHMİN YAP
# ─────────────────────────────────────────────
knn_preds = knn.predict(X_test_scaled)
rf_preds  = rf.predict(X_test_scaled)

# ─────────────────────────────────────────────
# 7. SONUÇLARI YAZDIR
# ─────────────────────────────────────────────
knn_acc = accuracy_score(y_test, knn_preds)
rf_acc  = accuracy_score(y_test, rf_preds)

print("\n========== SONUÇLAR ==========")
print(f"KNN Accuracy         : {knn_acc:.4f}")
print(f"Random Forest Accuracy: {rf_acc:.4f}")
print("==============================\n")

print("KNN Confusion Matrix:")
print(confusion_matrix(y_test, knn_preds))

print("\nRandom Forest Confusion Matrix:")
print(confusion_matrix(y_test, rf_preds))

# ─────────────────────────────────────────────
# 8. CONFUSION MATRIX GRAFİKLERİ
# ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

ConfusionMatrixDisplay.from_predictions(
    y_test, knn_preds, ax=axes[0], xticks_rotation=45
)
axes[0].set_title(f"KNN (Accuracy: {knn_acc:.4f})")

ConfusionMatrixDisplay.from_predictions(
    y_test, rf_preds, ax=axes[1], xticks_rotation=45
)
axes[1].set_title(f"Random Forest (Accuracy: {rf_acc:.4f})")

plt.tight_layout()
plt.savefig("confusion_matrices.png", dpi=150)
plt.show()
print("Grafik 'confusion_matrices.png' olarak kaydedildi.")

# ─────────────────────────────────────────────
# SONUÇ / CONCLUSION
# ─────────────────────────────────────────────
# NSL-KDD veri seti üzerinde KNN ve Random Forest algoritmaları karşılaştırıldı.
#
# Random Forest, KNN'e kıyasla genellikle daha yüksek accuracy elde eder.
# Bunun temel nedenleri:
#   - NSL-KDD'de farklı saldırı türleri arasındaki karar sınırları
#     doğrusal değildir; Random Forest bu non-linear ilişkileri
#     ağaç yapısı sayesinde daha iyi yakalar.
#   - Random Forest 100 ağacın oylamasıyla karar verir (ensemble),
#     bu sayede tek bir ağacın hatasına karşı dayanıklıdır.
#   - KNN, yüksek boyutlu verilerde (41 özellik) "curse of dimensionality"
#     sorunuyla karşılaşabilir: uzak komşular da "yakın" görünür hale gelir
#     ve sınıflandırma kalitesi düşer.
#   - KNN, test sırasında tüm train verisini taradığı için büyük veri
#     setlerinde (125k satır) daha yavaş çalışır.