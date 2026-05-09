import nltk
nltk.download('webtext')

from nltk.corpus import webtext
text = ' '.join(webtext.words('firefox.txt'))
print(len(text.split()))  # ~4000 كلمة بس ✅
import re, string

def Data_Cleaning(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text
import nltk
nltk.download('punkt')

def tokenize_sentences(text):
    tokens = nltk.word_tokenize(text)
    return tokens
from nltk import bigrams
from collections import Counter
import pandas as pd

def Bi_grams(tokens):
    bi = list(bigrams(tokens))
    return Counter(bi)

def prob_matrix(bigram_counts):
    df = pd.DataFrame(
        [(w1, w2, count) for (w1, w2), count in bigram_counts.items()],
        columns=['word', 'next_word', 'count']
    )
    return df
def predict_next(prev_word, df, top_n=5):
    filtered = df[df['word'] == prev_word]
    filtered = filtered.sort_values('count', ascending=False)
    return filtered['next_word'].head(top_n).tolist()
import streamlit as st

st.title("NLP Autofill")
user_input = st.text_input("اكتب كلمة:")

if user_input:
    last_word = user_input.strip().split()[-1]
    suggestions = predict_next(last_word, df)
    for s in suggestions:
        if st.button(s):
            st.write(f"اخترت: {s}")