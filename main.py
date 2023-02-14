import warnings
import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from torchtext.data import get_tokenizer
import re
import pickle
import streamlit as st
import pysrt
import seaborn as sns
import matplotlib.pyplot as plt
import webbrowser

HTML = "<.*?>"
TAG = "{.*?}"
LETTERS = "[^a-zA-Z\'.,!? ]"
TOKENIZER = get_tokenizer('spacy')
EMBEDDINGS = Word2Vec.load("word2vec_literature.model", mmap='r').wv
N_SPLITS = 20
LOG_REG = pickle.load(open('log_reg_model.pkl', 'rb'))
RAND_FORREST = pickle.load(open('rf_model.pkl', 'rb'))
CATBOOST = pickle.load(open('cb_model.pkl', 'rb'))
FINAL_LOG_REG = pickle.load(open('final_log_reg_model.pkl', 'rb'))


def clean_subs(subs):
    txt = re.sub(HTML, ' ', subs)
    txt = re.sub(TAG, ' ', txt)
    txt = re.sub(LETTERS, ' ', txt)
    return ' '.join(txt.lower().split())


def tokenize(text, tokenizer=TOKENIZER):
    return np.asarray(tokenizer(text))


def vectorize(splitted_tokens, embeddings=EMBEDDINGS):
    embeded_texts = np.zeros(300)
    for portion in splitted_tokens:
        text_embed = np.zeros(300)
        for token in portion:
            if token in embeddings:
                text_embed = text_embed + embeddings[token]
        embeded_texts = np.vstack((embeded_texts, text_embed))
    return embeded_texts[1:]


def split_vectorize(tokens, n_splits=N_SPLITS):
    return vectorize(np.array_split(tokens, n_splits))


warnings.filterwarnings('ignore')

st.title('English level estimator')
st.text('Please upload English subtitles for your movie.\n'
        'Then push the button and Machine Learning algorithms will tell you what\n'
        'English level does it take to understand the movie.')
st.markdown('You can download subs from https://subscene.com/.')
uploaded_file = st.file_uploader("**Upload subtitles file in *\*.srt* format.**", type='srt')
if uploaded_file:
    f = uploaded_file.read()
    try:
        subtitles = pysrt.from_string(f.decode('utf-16'))
    except UnicodeDecodeError:
        try:
            subtitles = pysrt.from_string(f.decode('utf-8'))
        except UnicodeDecodeError:
            try:
                subtitles = pysrt.from_string(f.decode('latin-1'))
            except UnicodeDecodeError:
                st.text('Please check encoding. It could be utf-16, utf-8 or latin-1')

predict_button = st.button('Estimate English level')
if predict_button:
    try:
        X = clean_subs(subtitles.text)
        X = tokenize(X)
        X = split_vectorize(X)
        # order of estimators DOES matter
        first_lvl_preds = np.empty((1,))
        first_lvl_preds = np.hstack((first_lvl_preds,
                                     LOG_REG.predict_proba(X).flatten()))
        first_lvl_preds = np.hstack((first_lvl_preds,
                                     RAND_FORREST.predict_proba(X).flatten()))
        first_lvl_preds = np.hstack((first_lvl_preds,
                                     CATBOOST.predict_proba(X).flatten()))
        first_lvl_preds = first_lvl_preds[1:].reshape(1, -1)
        predictions = pd.DataFrame({'Probability': FINAL_LOG_REG.predict_proba(first_lvl_preds)[0],
                                    'English Level': ['A2', 'B1', 'B2', 'C1']})
        msg = 'It seems that ' + predictions.loc[predictions['Probability'].idxmax(), 'English Level'] + \
              '-level would fit to watch this movie!'
        st.header(msg)
        st.text('Here is the diagram with probabilities of every English level for your movie.')
        st.text('A1 and C2 is not supported yet.')
        fig = plt.figure(figsize=(10, 4))
        sns.barplot(data=predictions, y='Probability', x='English Level')
        st.pyplot(fig)
    except NameError:
        st.text('Please upload subtitles first.')

disagree_button = st.button('I disagree with estimation')
if disagree_button:
    webbrowser.open('https://forms.yandex.ru/cloud/63e9400384227c817ffb5315/', new=2)
