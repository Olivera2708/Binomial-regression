from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pandas as pd
from features import get_features


data = pd.read_csv('data.csv')
articles = data['article']
y = [0 if category == "FactContext" else 1 for category in data['simple_user_need']]

X = get_features(articles)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

#Big difference in classes
print(data['simple_user_need'].value_counts())

model = LogisticRegression()
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=kfold)
print("Cross-validation scores:", scores)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy: {accuracy}")