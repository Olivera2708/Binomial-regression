from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_curve, auc
import pandas as pd
from features import get_features_basic, get_features_text_ent
from imblearn.over_sampling import SMOTETomek
from imblearn.under_sampling import RandomUnderSampler
import matplotlib.pyplot as plt


def draw_diagram(fpr, tpr, auc):
    fig = plt.figure()
    fig.canvas.manager.set_window_title('ROC')
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {auc:.2f})')
    plt.plot([0, 1], [0, 1], color='red', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right')
    plt.grid()
    plt.show()


def load_data1000():
    data = pd.read_csv('small_equal.csv')
    articles = data['article']
    y = [0 if category == "FactContext" else 1 for category in data['simple_user_need']]
    X = get_features_text_ent(articles)
    data = pd.concat([X, pd.DataFrame(y)], axis=1)
    cleaned_data = data.dropna()
    X = pd.DataFrame(cleaned_data.iloc[:, 60:-1])
    y = cleaned_data.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, shuffle=True)
    return X, y, X_train, X_test, y_train, y_test

def load_given_data():
    data = pd.read_csv('data.csv')
    articles = data['article']
    y = [0 if category == "FactContext" else 1 for category in data['simple_user_need']]
    X = get_features_text_ent(articles)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, shuffle=True)
    return X, y, X_train, X_test, y_train, y_test

def load_oversampling():
    data = pd.read_csv('data.csv')
    articles = data['article']
    y = [0 if category == "FactContext" else 1 for category in data['simple_user_need']]
    X = get_features_text_ent(articles)
    smote = SMOTETomek() #preporuka
    X, y = smote.fit_resample(X, y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, shuffle=True)
    return X, y, X_train, X_test, y_train, y_test

def load_undersampling():
    data = pd.read_csv('data.csv')
    articles = data['article']
    y = [0 if category == "FactContext" else 1 for category in data['simple_user_need']]
    X = get_features_text_ent(articles)
    undersampler = RandomUnderSampler()
    X, y = undersampler.fit_resample(X, y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, shuffle=True)
    return X, y, X_train, X_test, y_train, y_test


if __name__ == "__main__":
    X, y, X_train, X_test, y_train, y_test = load_data1000()

    #Big difference in classes
    # print(data['simple_user_need'].value_counts())

    model = LogisticRegression(max_iter=1000)
    kfold = KFold(n_splits=5, shuffle=True)
    scores = cross_val_score(model, X, y, cv=kfold)
    print("Cross-validation scores: ", scores)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")

    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    draw_diagram(fpr, tpr, roc_auc)