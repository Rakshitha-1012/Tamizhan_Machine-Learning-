# ----------------------------------------
# Project 1: Email Spam Detection
# ----------------------------------------

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Sample spam dataset
spam_data = pd.DataFrame({
    'text': ['Win money now!', 'Hi, how are you?', 'Free lottery ticket', 'Meeting at 10', 'Claim your prize'],
    'label': [1, 0, 1, 0, 1]  # 1=spam, 0=ham
})

tfidf = TfidfVectorizer()
X = tfidf.fit_transform(spam_data['text'])
y = spam_data['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

model1 = MultinomialNB()
model1.fit(X_train, y_train)
y_pred1 = model1.predict(X_test)
print("Spam Detection Accuracy:", accuracy_score(y_test, y_pred1))


# ----------------------------------------
# Project 2: Handwritten Digit Recognition (MNIST)
# ----------------------------------------

from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1,28,28,1)/255.0
x_test = x_test.reshape(-1,28,28,1)/255.0
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

model2 = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(100, activation='relu'),
    Dense(10, activation='softmax')
])

model2.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model2.fit(x_train, y_train, epochs=1, batch_size=32)  # Use more epochs for real training
print("MNIST Test Accuracy:", model2.evaluate(x_test, y_test)[1])


# ----------------------------------------
# Project 3: Loan Eligibility Predictor
# ----------------------------------------

data3 = {
    'Age': [25, 32, 47, 51, 62],
    'Income': [50000, 60000, 80000, 120000, 150000],
    'Education': ['Graduate', 'Not Graduate', 'Graduate', 'Graduate', 'Not Graduate'],
    'Credit_Score': [700, 650, 800, 720, 680],
    'Loan_Status': ['Y', 'N', 'Y', 'Y', 'N']
}
df3 = pd.DataFrame(data3)
df3['Education'] = df3['Education'].map({'Graduate':1, 'Not Graduate':0})
df3['Loan_Status'] = df3['Loan_Status'].map({'Y': 1, 'N': 0})
X3 = df3.drop('Loan_Status', axis=1)
y3 = df3['Loan_Status']
X_train3, X_test3, y_train3, y_test3 = train_test_split(X3, y3, test_size=0.2, random_state=42)
model3 = LogisticRegression()
model3.fit(X_train3, y_train3)
print("Loan Predictor Accuracy:", model3.score(X_test3, y_test3))


# ----------------------------------------
# Project 4: Fake News Detection
# ----------------------------------------

from sklearn.linear_model import PassiveAggressiveClassifier

news_df = pd.DataFrame({
    'text': ['The economy is growing', 'Aliens landed in the US', 'Government announces new policy'],
    'label': [1, 0, 1]
})
X4 = tfidf.fit_transform(news_df['text'])
y4 = news_df['label']
X_train4, X_test4, y_train4, y_test4 = train_test_split(X4, y4, test_size=0.2)
model4 = PassiveAggressiveClassifier()
model4.fit(X_train4, y_train4)
print("Fake News Accuracy:", model4.score(X_test4, y_test4))


# ----------------------------------------
# Project 5: Movie Recommendation System
# ----------------------------------------

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

ratings = pd.DataFrame({
    'User1': [5, 4, np.nan, 1],
    'User2': [4, np.nan, 4, 1],
    'User3': [1, 1, 5, 4]
}, index=['Movie1', 'Movie2', 'Movie3', 'Movie4'])

sim_matrix = cosine_similarity(ratings.fillna(0).T)
print("User Similarity Matrix:
", sim_matrix)


# ----------------------------------------
# Project 6: Stock Price Prediction (LSTM)
# ----------------------------------------

import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Dummy time-series data
prices = np.array([100, 102, 101, 105, 107, 110, 108, 111])
X6, y6 = [], []
for i in range(len(prices) - 3):
    X6.append(prices[i:i+3])
    y6.append(prices[i+3])
X6 = np.array(X6).reshape(-1, 3, 1)
y6 = np.array(y6)

model6 = Sequential([
    LSTM(50, input_shape=(3,1)),
    Dense(1)
])
model6.compile(optimizer='adam', loss='mse')
model6.fit(X6, y6, epochs=100, verbose=0)
pred6 = model6.predict(X6)
plt.plot(y6, label="Actual")
plt.plot(pred6, label="Predicted")
plt.legend()
plt.title("Stock Price Prediction")
plt.show()


# ----------------------------------------
# Project 7: Emotion Detection from Text
# ----------------------------------------

emotion_data = pd.DataFrame({
    'text': ['I am happy today', 'This is terrible', 'I am very excited', 'I feel so sad'],
    'emotion': ['happy', 'angry', 'happy', 'sad']
})

X7 = tfidf.fit_transform(emotion_data['text'])
y7 = emotion_data['emotion']
model7 = LogisticRegression()
model7.fit(X7, y7)
print("Emotion Prediction:", model7.predict(tfidf.transform(["I am thrilled"])))
