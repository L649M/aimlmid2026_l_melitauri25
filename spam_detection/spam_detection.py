import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score



# load dataset
df = pd.read_csv("../data/l_melitauri25_31698.csv")

# show first rows
print(df.head())

# show general info
print(df.info())

# separate features and label
X = df.drop("is_spam", axis=1)
y = df["is_spam"]

print("Features (X):")
print(X.head())

print("\nLabel (y):")
print(y.head())

# split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

print("Training set size:", X_train.shape)
print("Test set size:", X_test.shape)

# create logistic regression model
model = LogisticRegression(max_iter=1000)

# train the model
model.fit(X_train, y_train)

print("Model trained successfully")


# show model coefficients
coefficients = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_[0]
})

print("\nModel Coefficients:")
print(coefficients)

# make predictions on test data
y_pred = model.predict(X_test)

print("Predictions (first 10):")
print(y_pred[:10])


# confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

# accuracy score
accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy:", accuracy)



# function to extract features from raw email text
def extract_features_from_text(text):
    words = len(text.split())
    links = text.lower().count("http")
    capital_words = sum(1 for w in text.split() if w.isupper())

    spam_words = ["free", "win", "money", "offer", "click"]
    spam_word_count = sum(text.lower().count(w) for w in spam_words)

    return words, links, capital_words, spam_word_count



# function to classify a new email based on extracted features
def classify_email(words, links, capital_words, spam_word_count):
    data = [[words, links, capital_words, spam_word_count]]
    prediction = model.predict(data)[0]

    if prediction == 1:
        return "Spam"
    else:
        return "Legitimate"

# example spam email
spam_result = classify_email(
    words=120,
    links=5,
    capital_words=20,
    spam_word_count=8
)


# manual spam email text
spam_text = """
WIN WIN WIN FREE MONEY NOW!!!
CLICK HERE http://spam.com http://spam.com
LIMITED OFFER!!! FREE FREE FREE!!!
ACT NOW AND GET MONEY!!!
"""


spam_features = extract_features_from_text(spam_text)
spam_text_result = classify_email(*spam_features)

print("\nSpam email TEXT classification result:", spam_text_result)


# manual legitimate email text
legit_text = """
Hi John,

I hope you are doing well.
Please find attached the report from our last meeting.

Best regards,
Lasha
"""

legit_features = extract_features_from_text(legit_text)
legit_text_result = classify_email(*legit_features)

print("Legitimate email TEXT classification result:", legit_text_result)



print("\nSpam email classification result:", spam_result)

# example legitimate email
legit_result = classify_email(
    words=300,
    links=0,
    capital_words=1,
    spam_word_count=0
)

print("Legitimate email classification result:", legit_result)

# visualization 1: class distribution
plt.figure()
df["is_spam"].value_counts().plot(kind="bar")
plt.xlabel("Email Class")
plt.ylabel("Count")
plt.title("Spam vs Legitimate Email Distribution")
plt.xticks([0, 1], ["Legitimate", "Spam"], rotation=0)
plt.tight_layout()
plt.savefig("class_distribution.png")
plt.show()

# visualization 2: confusion matrix
plt.figure()
plt.imshow(cm)
plt.title("Confusion Matrix")
plt.colorbar()

plt.xticks([0, 1], ["Legitimate", "Spam"])
plt.yticks([0, 1], ["Legitimate", "Spam"])

for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j], ha="center", va="center")

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()
