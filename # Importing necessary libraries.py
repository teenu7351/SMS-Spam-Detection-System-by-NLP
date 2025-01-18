# Importing necessary libraries
import pandas as pd
import numpy as np
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Downloading NLTK resources (if you haven't already)
nltk.download('stopwords')
nltk.download('punkt')

# Step 1: Load Dataset (Assuming 'spam.csv' dataset)
# The dataset should have two columns: 'label' (spam/ham) and 'message' (SMS content)
data = pd.read_csv('spam.csv', encoding='latin-1')

# Checking the first few rows of the dataset
print(data.head())

# Step 2: Preprocessing the data
# Removing unnecessary columns
data = data[['v1', 'v2']]  # 'v1' = label, 'v2' = message
data.columns = ['label', 'message']

# Text Preprocessing
# Removing stopwords and tokenizing text using NLTK
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    # Tokenize text and remove stopwords
    words = nltk.word_tokenize(text)
    return ' '.join([word.lower() for word in words if word.isalpha() and word.lower() not in stop_words])

# Apply preprocessing to the messages
data['message'] = data['message'].apply(preprocess_text)

# Step 3: Feature Extraction (TF-IDF)
# Convert text into numerical features using TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(data['message']).toarray()

# Step 4: Encoding labels (spam=1, ham=0)
y = data['label'].map({'ham': 0, 'spam': 1})

# Step 5: Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Model Training using Naive Bayes
model = MultinomialNB()
model.fit(X_train, y_train)

# Step 7: Making predictions on the test data
y_pred = model.predict(X_test)

# Step 8: Evaluating the Model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy*100:.2f}%")

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

# Visualizing Confusion Matrix
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Step 9: Example Prediction
sample_message = ["Free entry in a $1000 prize draw!!!"]
sample_message_processed = [preprocess_text(msg) for msg in sample_message]
sample_features = vectorizer.transform(sample_message_processed).toarray()
sample_prediction = model.predict(sample_features)

if sample_prediction[0] == 1:
    print("\nThe message is SPAM.")
else:
    print("\nThe message is HAM.")
