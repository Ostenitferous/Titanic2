import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

train_data = pd.read_csv("train.csv")

train_data = train_data.drop(["Name", "Ticket", "Cabin", "Embarked"], axis=1)

train_data["Age"].fillna(train_data["Age"].median(), inplace=True)

train_data["Sex"] = pd.factorize(train_data["Sex"])[0]

X_train = train_data.drop(["Survived", "PassengerId"], axis=1)
y_train = train_data["Survived"]

model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
model.fit(X_train, y_train)

survival_rate_train = train_data["Survived"].mean()
print("Train veri setinde hayatta kalma yüzdesi:", survival_rate_train)

test_data = pd.read_csv("test.csv")

test_data = test_data.drop(["Name", "Ticket", "Cabin", "Embarked"], axis=1)

test_data["Age"].fillna(test_data["Age"].median(), inplace=True)
test_data["Fare"].fillna(test_data["Fare"].median(), inplace=True)

test_data["Sex"] = pd.factorize(test_data["Sex"])[0]

predictions = model.predict(test_data.drop("PassengerId", axis=1))
probabilities = model.predict_proba(test_data.drop("PassengerId", axis=1))[:, 1]

survival_rate_test = predictions.mean()
print("Test veri setinde tahmini hayatta kalma yüzdesi:", survival_rate_test)

output = pd.DataFrame({"PassengerId": test_data["PassengerId"], "Survived": predictions})
output.to_csv("predictions.csv", index=False)

test_data_with_results = pd.read_csv("gender_submission.csv")
accuracy = accuracy_score(test_data_with_results["Survived"], predictions)
print("Modelin doğruluğu:", accuracy)
