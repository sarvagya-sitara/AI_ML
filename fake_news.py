import os
import re
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score


print("Loading Fake News Detector...")
file_name = "news.csv"

if not os.path.exists(file_name):
    print(f"'{file_name}' not found!")
    exit()

data = pd.read_csv(file_name)
print(f"Dataset loaded: {len(data)} rows")


data = data.dropna(subset=["text", "label"])
data["label"] = data["label"].astype(str).str.upper().str.strip()
data = data[data["label"].isin(["REAL", "FAKE"])]
data["label"] = data["label"].map({"REAL": 1, "FAKE": 0})


data = data.drop_duplicates(subset="text")


print("\nLabel Distribution BEFORE balancing:")
print(data["label"].value_counts())


real = data[data["label"] == 1]
fake = data[data["label"] == 0]

min_size = min(len(real), len(fake))

data = pd.concat([
    real.sample(min_size, random_state=42),
    fake.sample(min_size, random_state=42)
])


data = data.sample(frac=1, random_state=42).reset_index(drop=True)

print("\nLabel Distribution AFTER balancing:")
print(data["label"].value_counts())


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


data["text"] = data["text"].apply(clean_text)


X = data["text"]
y = data["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


vectorizer = TfidfVectorizer(
    stop_words="english",
    max_df=0.85,
    ngram_range=(1,2),
    min_df=2
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)


model = MultinomialNB(alpha=0.5)
model.fit(X_train_vec, y_train)


y_pred = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)

print(f"\nAccuracy: {accuracy:.2%}")
print("="*60)


def predict_news(news_text):
    cleaned = clean_text(news_text)
    vector = vectorizer.transform([cleaned])
    prediction = model.predict(vector)[0]
    probability = model.predict_proba(vector)[0]
    confidence = max(probability)

    if prediction == 1:
        return f"REAL (confidence: {confidence:.1%})"
    else:
        return f"FAKE (confidence: {confidence:.1%})"


print("\nTesting:")
print(predict_news("ISRO launches new satellite successfully"))
print(predict_news("Aliens invade India and control government"))
print(predict_news("Government announces new education policy"))


print("\nEnter news (type 'quit' to exit):")

while True:
    user_input = input("\nNews: ").strip()

    if user_input.lower() in ['quit', 'exit', 'q']:
        print("Exiting...")
        break

    if not user_input:
        print("Enter something")
        continue

    print(predict_news(user_input))
