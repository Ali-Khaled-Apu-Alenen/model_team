import nltk
import re
import string
import pandas as pd
import streamlit as st
from nltk import bigrams
from collections import Counter

# تحميل البيانات
nltk.download('webtext', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

# ===== الخطوة 1: جلب النص =====
from nltk.corpus import webtext
raw_text = ' '.join(webtext.words('firefox.txt'))

# ===== الخطوة 2: تنظيف النص =====
def Data_Cleaning(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

# ===== الخطوة 3: تقطيع لكلمات =====
def tokenize_sentences(text):
    tokens = nltk.word_tokenize(text)
    return tokens

# ===== الخطوة 4: بناء Bigrams =====
def Bi_grams(tokens):
    bi = list(bigrams(tokens))
    return Counter(bi)

# ===== الخطوة 5: بناء DataFrame =====
def prob_matrix(bigram_counts):
    df = pd.DataFrame(
        [(w1, w2, count) for (w1, w2), count in bigram_counts.items()],
        columns=['word', 'next_word', 'count']
    )
    return df

# ===== الخطوة 6: التنبؤ =====
def predict_next(prev_word, df, top_n=5):
    filtered = df[df['word'] == prev_word]
    filtered = filtered.sort_values('count', ascending=False)
    return filtered['next_word'].head(top_n).tolist()

# ===== تشغيل كل الخطوات =====
cleaned = Data_Cleaning(raw_text)
tokens = tokenize_sentences(cleaned)
bigram_counts = Bi_grams(tokens)
df = prob_matrix(bigram_counts)

# ===== الواجهة =====
st.title("NLP Autofill")
user_input = st.text_input("اكتب كلمة:")

if user_input:
    last_word = user_input.strip().split()[-1].lower()
    suggestions = predict_next(last_word, df)
    
    if suggestions:
        st.write("الاقتراحات:")
        for s in suggestions:
            if st.button(s):
                st.write(f"اخترت: {s}")
    else:
        st.write("مافيش اقتراحات لهذه الكلمة")
