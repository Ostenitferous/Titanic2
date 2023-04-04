import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Train veri setini yükle
train_data = pd.read_csv("train.csv")

# Gereksiz sütunları kaldır
train_data = train_data.drop(["Name", "Ticket", "Cabin", "Embarked"], axis=1)

# Eksik verileri doldur
train_data["Age"].fillna(train_data["Age"].median(), inplace=True)

# Kategorik verileri sayısala çevir
train_data["Sex"] = pd.factorize(train_data["Sex"])[0]

# Bağımsız ve bağımlı değişkenleri ayır
X_train = train_data.drop(["Survived", "PassengerId"], axis=1)
y_train = train_data["Survived"]

# Modeli oluştur ve eğit
model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
model.fit(X_train, y_train)

# Train veri setindeki hayatta kalma yüzdesini hesapla ve yazdır
survival_rate_train = train_data["Survived"].mean()
print("Train veri setinde hayatta kalma yüzdesi:", survival_rate_train)

# Test veri setini yükle
test_data = pd.read_csv("test.csv")

# Gereksiz sütunları kaldır
test_data = test_data.drop(["Name", "Ticket", "Cabin", "Embarked"], axis=1)

# Eksik verileri doldur
test_data["Age"].fillna(test_data["Age"].median(), inplace=True)
test_data["Fare"].fillna(test_data["Fare"].median(), inplace=True)

# Kategorik verileri sayısala çevir
test_data["Sex"] = pd.factorize(test_data["Sex"])[0]

# Test verisi üzerinde tahmin yap
predictions = model.predict(test_data.drop("PassengerId", axis=1))
probabilities = model.predict_proba(test_data.drop("PassengerId", axis=1))[:, 1]

# Test veri setindeki tahmini hayatta kalma yüzdesini hesapla ve yazdır
survival_rate_test = predictions.mean()
print("Test veri setinde tahmini hayatta kalma yüzdesi:", survival_rate_test)

# Sonuçları dosyaya yazdır
output = pd.DataFrame({"PassengerId": test_data["PassengerId"], "Survived": predictions})
output.to_csv("predictions.csv", index=False)

# Test veri setindeki gerçek sonuçları yükle ve modelin doğruluğunu hesapla
test_data_with_results = pd.read_csv("gender_submission.csv")
accuracy = accuracy_score(test_data_with_results["Survived"], predictions)
print("Modelin doğruluğu:", accuracy)
