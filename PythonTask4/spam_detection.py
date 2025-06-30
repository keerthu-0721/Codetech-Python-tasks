import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. Simulate a Dataset (for demonstration purposes) ---
# In a real scenario, you would load your dataset from a CSV, database, etc.
data = {
    'email_content': [
        "Congratulations! You've won a free iPhone. Click here!",
        "Meeting reminder for tomorrow at 10 AM.",
        "URGENT: Your account has been compromised. Verify now!",
        "Hi team, please review the latest project updates.",
        "Claim your prize money now! Limited time offer.",
        "Regarding the quarterly financial report.",
        "VIAGRA! Best deals on medication.",
        "Hello, just checking in on the task progress.",
        "Free money! No strings attached. Act fast!",
        "Your order has been shipped. Tracking number inside.",
        "Exclusive offer: Get 50% off on all products!",
        "Project deadline approaching. Let's sync up.",
        "You are selected for a secret millionaire program!",
        "Review of last week's performance metrics.",
        "Win a luxury vacation! Enter now.",
    ],
    'label': [
        'spam', 'ham', 'spam', 'ham', 'spam', 'ham', 'spam', 'ham', 'spam', 'ham',
        'spam', 'ham', 'spam', 'ham', 'spam'
    ]
}
df = pd.DataFrame(data)

print("--- Dataset Overview ---")
print(df.head())
print("\nLabel distribution:")
print(df['label'].value_counts())
print("-" * 30)

# --- 2. Data Preprocessing: Feature Extraction ---
# Convert text data into numerical features using CountVectorizer (Bag-of-Words)
# This counts the occurrences of each word in each email.
vectorizer = CountVectorizer(stop_words='english', lowercase=True)
X = vectorizer.fit_transform(df['email_content'])
y = df['label']

print(f"\nVocabulary size: {len(vectorizer.vocabulary_)}")
# print(f"Sample feature names: {vectorizer.get_feature_names_out()[:10]}") # Uncomment to see features
print("-" * 30)

# --- 3. Split Data into Training and Testing Sets ---
# We split the data to evaluate the model's performance on unseen data.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

print(f"\nTraining samples: {X_train.shape[0]}")
print(f"Testing samples: {X_test.shape[0]}")
print("-" * 30)

# --- 4. Model Training ---
# Using Multinomial Naive Bayes, a common classifier for text classification.
model = MultinomialNB()
print("\nTraining the model...")
model.fit(X_train, y_train)
print("Model training complete.")
print("-" * 30)

# --- 5. Make Predictions ---
print("\nMaking predictions on the test set...")
y_pred = model.predict(X_test)
print("Predictions made.")
print("-" * 30)

# --- 6. Model Evaluation ---
print("\n--- Model Evaluation ---")
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred, labels=['ham', 'spam']) # Specify labels for consistent order
print(cm)

# Visualize Confusion Matrix
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Predicted Ham', 'Predicted Spam'], yticklabels=['Actual Ham', 'Actual Spam'])
plt.title('Confusion Matrix for Spam Detection')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.show()
print("-" * 30)

# --- 7. Test with New Data ---
print("\n--- Testing with New, Unseen Emails ---")
new_emails = [
    "You have won a lottery! Claim your prize now.",
    "Hello John, can we schedule a meeting next week?",
    "Your credit card has been suspended. Click to reactivate.",
    "Just a friendly reminder about your appointment.",
    "Huge discount on medicines, buy now!",
    "Important update: New policy changes.",
]

# Transform new emails using the SAME vectorizer fitted on training data
new_emails_transformed = vectorizer.transform(new_emails)
new_predictions = model.predict(new_emails_transformed)

for email, prediction in zip(new_emails, new_predictions):
    print(f"Email: '{email}'\nPredicted: {prediction.upper()}\n")

print("Predictive model demonstration complete.")