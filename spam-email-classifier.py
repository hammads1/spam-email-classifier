# Install dependencies if you haven't already
# pip install pandas scikit-learn

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# 1. Sample dataset (replace with your dataset)
data = {
    'email': [
        "Congratulations! You've won a $1000 gift card. Click here to claim now.",
        "Hi Hammad, can we meet tomorrow to discuss the project?",
        "Limited time offer! Buy now and get 50% off your purchase.",
        "Dear student, your assignment is due next Monday.",
        "You have been selected for a free lottery. Claim your prize!",
        "Please find attached the report for your review."
    ],
    'label': [1, 0, 1, 0, 1, 0]  # 1 = spam, 0 = not spam
}

df = pd.DataFrame(data)

# 2. Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    df['email'], df['label'], test_size=0.3, random_state=42
)

# 3. Convert text to numerical features
vectorizer = TfidfVectorizer()
X_train_vectors = vectorizer.fit_transform(X_train)
X_test_vectors = vectorizer.transform(X_test)

# 4. Train the classifier
classifier = MultinomialNB()
classifier.fit(X_train_vectors, y_train)

# 5. Make predictions
y_pred = classifier.predict(X_test_vectors)

# 6. Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# 7. Test on new emails
new_emails = [
    "Win a brand new car by entering our contest now!",
    "Can you send me the meeting notes from yesterday?"
]
new_vectors = vectorizer.transform(new_emails)
predictions = classifier.predict(new_vectors)

for email, pred in zip(new_emails, predictions):
    label = "Spam" if pred == 1 else "Not Spam"
    print(f"Email: {email}\nPrediction: {label}\n")
