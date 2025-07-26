import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

# Load the dataset
df = pd.read_csv("IMDB Dataset.csv")
df = df[['review', 'sentiment']].dropna()

# Convert target to binary
df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})

# Vectorize text data
vectorizer = CountVectorizer(stop_words='english', max_features=10000)
X = vectorizer.fit_transform(df['review'])
y = df['sentiment']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Naive Bayes
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)
nb_pred = nb_model.predict(X_test)

# Train Logistic Regression
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)
log_pred = log_model.predict(X_test)

# Print accuracies
print("Naive Bayes Accuracy:", accuracy_score(y_test, nb_pred))
print("Logistic Regression Accuracy:", accuracy_score(y_test, log_pred))

# Save models and vectorizer
joblib.dump(nb_model, "nb_model.pkl")
joblib.dump(log_model, "log_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")
