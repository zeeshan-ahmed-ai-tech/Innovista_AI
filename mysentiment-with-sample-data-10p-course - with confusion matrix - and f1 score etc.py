from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import pandas as pd
import matplotlib.pyplot as plt

# User test data
statements = [
    "I feel great today",
    "I dont feel great today",
    "I dont feel good or bad today",
    "demonstrated exceptional dedication and commitment, going above and beyond to ensure seamless operations",
    "i am really feeling great today, but i don't know how the day will turn out"
]

# Load data from CSV file
#file_path = 'data.csv'
file_path = 'data-trained.csv'
df = pd.read_csv(file_path, sep=',',
                 encoding='ISO-8859-1',
                 dtype={'Sentiment': str, 'Text': str},
                 header=None,
                 names=["Sentiment", "Text"])

# Convert 'Sentiment' to numeric
df['Sentiment'] = pd.to_numeric(df['Sentiment'], errors='coerce')

# Drop rows with NaN values in the target variable 'Sentiment'
df = df.dropna(subset=['Sentiment'])

# Vectorize the entire dataset
vectorizer = TfidfVectorizer()
X_tfidf = vectorizer.fit_transform(df['Text'])

# Train the classifier on the entire dataset
classifier = MultinomialNB()
classifier.fit(X_tfidf, df['Sentiment'])

# Vectorize the new statements
new_statements_tfidf = vectorizer.transform(statements)

# Make predictions on the new statements
new_statements_predictions = classifier.predict(new_statements_tfidf)

# Print or analyze the predictions
for statement, prediction in zip(statements, new_statements_predictions):
    print(f"Statement: {statement}")
    print(f"Predicted Sentiment: {prediction}\n")

# Display results
print(df[['Text', 'Sentiment']])

num_rows = df.shape[0]
print("Number of rows in the DataFrame:", num_rows)

# Count the occurrences of each sentiment label
sentiment_counts = df['Sentiment'].value_counts()

# Display the counts
print("Sentiment Counts:")
print(sentiment_counts)

print("4 means positive sentiment, 0 means negative sentiment")

# Generate confusion matrix
actual_labels = df['Sentiment'].values  # Replace with actual test labels if available
predicted_labels = classifier.predict(X_tfidf)  # Use the classifier to predict on the training data

# Compute confusion matrix
cm = confusion_matrix(actual_labels, predicted_labels)

# Compute evaluation metrics
accuracy = accuracy_score(actual_labels, predicted_labels)
precision = precision_score(actual_labels, predicted_labels, pos_label=4)
recall = recall_score(actual_labels, predicted_labels, pos_label=4)
f1 = f1_score(actual_labels, predicted_labels, pos_label=4)

# Display confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.show()

# Print evaluation metrics
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")
