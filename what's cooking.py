
'''import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the dataset
train_df = pd.read_json("C:/Users/DELL/PycharmProjects/pythonProject4/train.json/train.json")
print(train_df.head())
# Join the list of ingredients into strings
train_df['ingredients_str'] = train_df['ingredients'].apply(lambda x: ' '.join(x))

# Separate features and target variable
X = train_df['ingredients_str']
y = train_df['cuisine']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer()

# Fit and transform the training data
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)

# Initialize and train the classifier
clf = LogisticRegression(max_iter=1000, random_state=42)
clf.fit(X_train_tfidf, y_train)

# Transform the testing data
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Make predictions on the testing set
y_pred = clf.predict(X_test_tfidf)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Accuracy: {:.2f}".format(accuracy))'''



import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# Load the dataset
train_df = pd.read_json("C:/Users/DELL/PycharmProjects/pythonProject4/train.json/train.json")

# Join the list of ingredients into strings
train_df['ingredients_str'] = train_df['ingredients'].apply(lambda x: ' '.join(x))

# Separate features and target variable
X = train_df['ingredients_str']
y = train_df['cuisine']

# Initialize TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer()

# Fit and transform the training data
X_tfidf = tfidf_vectorizer.fit_transform(X)

# Initialize LabelEncoder
label_encoder = LabelEncoder()

# Encode the target variable
y_encoded = label_encoder.fit_transform(y)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y_encoded, test_size=0.2, random_state=42)

# Initialize and train the classifier
clf = LogisticRegression(max_iter=1000, random_state=42)
clf.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = clf.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
