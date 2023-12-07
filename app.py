# Import packages
pip install streamlit-tags
import streamlit as st
from streamlit_tags import st_tags
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
import re
import codecs
from hazm import Normalizer
import ast
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
# -------------------------------------------------------------------------------
nmz = Normalizer()
nltk.download("punkt")
nltk.download("stopwords")
en_stops = set(stopwords.words("english"))
dmd_df = pd.read_csv('dmd_df.csv')
prd_df = pd.read_csv('prd_df.csv')
# Import persian stops
fa_stops = sorted(list(set([nmz.normalize(w) for w in codecs.open(
    'persian.txt', encoding='utf-8').read().split('\n') if w])))

specific_stops = ['جمله', 'سیستم', 'تولید', 'دستگاه', 'طراحی', 'شرکت', 'ساخت', 'مخصوص',
                  'مصرف', 'کشور', 'خروجی', 'کیفیت', 'کاربرد', 'ارائه', 'کار', 'موجود', 'قطعه', 'سال']


def tokenize(df, columns):
    for c in columns:
        df = df.drop('tokenized_' + str(c), axis=1, errors='ignore')
        df.insert(loc=df.shape[1], column='tokenized_' + str(c), value=None)
        # Tokenize the Farsi dataset
        for i in range(df.shape[0]):
            tokenized_dataset = []
            # replace punctuation with space
            t = re.sub(r'[^\w\s]', ' ', str(df[c][i]))
            t = [nmz.normalize(t)]
            for w in t:
                tokens = nltk.word_tokenize(w)
                filtered_words = [word for word in tokens if (word not in fa_stops) & (
                    word not in en_stops) & (word not in specific_stops) & (not word.isdigit())]
                tokenized_dataset.append(filtered_words)
            df['tokenized_' + str(c)][i] = tokenized_dataset[0]
    return df


# -------------------------------------------------------------------------------
# title and subtitle
st.title('Technomart Matching Demo')
st.text('Maching demand and supply of Iran National Technomart')
# Demmender or supplier
input_type = st.radio(
    "Are you supplier or demmender?",
    key="visibility",
    options=["Supplier", "Demander"]
)
# Input Title, Description and Keywords
user_input_title = st.text_input(
    "Enter Title:", placeholder='Text Recommneder System', max_chars=100)
user_input_description = st.text_area(
    "Enter Description:", placeholder='I nee a text recommneder system', max_chars=2000, height=200)
keywords = []
user_input_keywords = st_tags(
    label="Add keywords:",
    text="Press enter to add",
    value=keywords,
    key="tag_input",
)
# --------------------------------------------------------------------------------

# Choose Algorithm, Number and Sort Type
col1, col2, col3 = st.columns(3)
with col1:
    algo = st.selectbox('Algorithm',
                        ('Cosine Similarity', 'Jaccard Similarity', 'LDA', 'Word2Vec'))

with col2:
    item_number = st.selectbox('Number',
                               set(range(1, 10)))
st.text("")
st.text("")
# # -------------------------------------------------------------------------------
# # RUN Button and Result
# button_id = st.button("Run", key="my_button", type='primary',
#                       use_container_width=True)
# if button_id:
#     js_code = """
#         <script>
#             const button = document.querySelector('[data-baseweb="button"]');

#             button.addEventListener('click', function() {
#                 button.style.backgroundColor = 'red';
#             });
#         </script>
#     """
#     st.markdown(js_code, unsafe_allow_html=True)
#     # --------------------------------------------------------------------------------
#     # Supplier
#     if input_type == 'Supplier':
#         prd_df = pd.DataFrame({'prd_urlIdentifier': 'PRD--1', 'prd_title': [user_input_title], 'prd_description': [user_input_description], 'prd_key_words': str([user_input_keywords])
#                                })
#         tokenized_prd_df = tokenize(
#             prd_df, ['prd_title', 'prd_description', 'prd_key_words'])
#         tokenized_dmd_df = pd.read_csv('tokenized_dmd_df.csv', converters={'tokenized_dmd_title': ast.literal_eval, 'tokenized_dmd_description': ast.literal_eval, 'tokenized_dmd_key_words': ast.literal_eval}
#                                        )

#         most_similar_dmd_for_prd_df = pd.DataFrame(
#             {'prd': prd_df['prd_urlIdentifier']})
#         for c in ['title', 'description', 'key_words']:
#             # Create lists of tokenized title for dmd and prd
#             dmd_token = [' '.join(tokens)
#                          for tokens in tokenized_dmd_df['tokenized_dmd_' + str(c)]]
#             prd_token = [' '.join(tokens)
#                          for tokens in tokenized_prd_df['tokenized_prd_' + str(c)]]
#         # Initialize a TF-IDF vectorizer
#             tfidf_vectorizer = TfidfVectorizer()

#             # Combine all title into one list
#             all_token = dmd_token + prd_token

#             # Fit and transform the TF-IDF vectorizer on all title
#             tfidf_matrix = tfidf_vectorizer.fit_transform(all_token)

#             # Calculate similarity between dmd and prd
#             if algo == 'Cosine Similarity':
#                 similarity_matrix = cosine_similarity(
#                     tfidf_matrix[:len(dmd_token)], tfidf_matrix[len(dmd_token):])
#                 d = {'title': 0.2, 'description': 0.35, 'key_words': 0.4}
#                 matching_results = pd.DataFrame(
#                     similarity_matrix * d[c], index=tokenized_dmd_df['dmd_urlIdentifier'], columns=tokenized_prd_df['prd_urlIdentifier'])
#             elif algo == 'Jaccard Similarity':
#                 similarity_matrix = tfidf_matrix[:len(dmd_token)].dot(
#                     tfidf_matrix[len(dmd_token):].T)
#                 d = {'title': 0.6, 'description': 0.4, 'key_words': 0.4}
#                 matching_results = pd.DataFrame(similarity_matrix.toarray(
#                 ) * d[c], index=tokenized_dmd_df['dmd_urlIdentifier'], columns=tokenized_prd_df['prd_urlIdentifier'])

#             # Find the most similar dmd for each prd
#             most_similar_dmd_for_prd = {}
#             for prd in prd_df['prd_urlIdentifier']:
#                 sorted_dmd = matching_results[prd].sort_values(ascending=False)
#                 # Exclude the prd herself
#                 most_similar_dmd = dict(sorted_dmd[:15])
#                 most_similar_dmd_for_prd[prd] = most_similar_dmd

#             # Create DataFrames to display the results
#             most_similar_dmd_for_prd_df = pd.merge(most_similar_dmd_for_prd_df, pd.DataFrame(
#                 most_similar_dmd_for_prd.items(), columns=['prd', 'Most Similar dmd ' + str(c)]))

#         most_similar_dmd_for_prd_df['total'] = most_similar_dmd_for_prd_df.apply(
#             lambda x: dict(
#                 sum(
#                     map(Counter, x.iloc[1:4].apply(lambda y: dict(y))), start=Counter()
#                 )
#             ), axis=1)

#         df = pd.DataFrame(most_similar_dmd_for_prd_df['total'].tolist()[0].items(), columns=[
#                           'ID', 'Values']).sort_values('Values', ascending=False).reset_index(drop=True).iloc[:item_number, :]
#         df.index += 1
#         df = pd.merge(df, tokenized_dmd_df[['dmd_urlIdentifier', 'dmd_title',
#                                             'dmd_key_words']], left_on='ID', right_on='dmd_urlIdentifier').drop('dmd_urlIdentifier', axis=1).rename(columns={'dmd_title': 'Title', 'dmd_key_words': 'keywords'})
#         df['Links'] = df['ID'].apply(
#             lambda r: f'<a href="https://techmart.ir/demand/view/{r}">Link</a>')

#     # ----------------------------------------------------------------------------------
#     # Demander
#     elif input_type == 'Demander':
#         dmd_df = pd.DataFrame({'dmd_urlIdentifier': 'dmd--1', 'dmd_title': [user_input_title], 'dmd_description': [user_input_description], 'dmd_key_words': str([user_input_keywords])
#                                })
#         tokenized_dmd_df = tokenize(
#             dmd_df, ['dmd_title', 'dmd_description', 'dmd_key_words'])
#         tokenized_prd_df = pd.read_csv('tokenized_prd_df.csv', converters={'tokenized_prd_title': ast.literal_eval, 'tokenized_prd_description': ast.literal_eval, 'tokenized_prd_key_words': ast.literal_eval}
#                                        )

#         most_similar_prd_for_dmd_df = pd.DataFrame(
#             {'dmd': dmd_df['dmd_urlIdentifier']})
#         for c in ['title', 'description', 'key_words']:
#             # Create lists of tokenized title for prd and dmd
#             prd_token = [' '.join(tokens)
#                          for tokens in tokenized_prd_df['tokenized_prd_' + str(c)]]
#             dmd_token = [' '.join(tokens)
#                          for tokens in tokenized_dmd_df['tokenized_dmd_' + str(c)]]
#         # Initialize a TF-IDF vectorizer
#             tfidf_vectorizer = TfidfVectorizer()

#             # Combine all title into one list
#             all_token = prd_token + dmd_token

#             # Fit and transform the TF-IDF vectorizer on all title
#             tfidf_matrix = tfidf_vectorizer.fit_transform(all_token)

#             # Calculate similarity between prd and dmd
#             if algo == 'Cosine Similarity':
#                 similarity_matrix = cosine_similarity(
#                     tfidf_matrix[:len(prd_token)], tfidf_matrix[len(prd_token):])
#                 d = {'title': 0.2, 'description': 0.35, 'key_words': 0.4}
#                 matching_results = pd.DataFrame(
#                     similarity_matrix * d[c], index=tokenized_prd_df['prd_urlIdentifier'], columns=tokenized_dmd_df['dmd_urlIdentifier'])
#             elif algo == 'Jaccard Similarity':
#                 similarity_matrix = tfidf_matrix[:len(prd_token)].dot(
#                     tfidf_matrix[len(prd_token):].T)
#                 d = {'title': 0.6, 'description': 0.4, 'key_words': 0.4}
#                 matching_results = pd.DataFrame(similarity_matrix.toarray(
#                 ) * d[c], index=tokenized_prd_df['prd_urlIdentifier'], columns=tokenized_dmd_df['dmd_urlIdentifier'])

#             # Find the most similar prd for each dmd
#             most_similar_prd_for_dmd = {}
#             for dmd in dmd_df['dmd_urlIdentifier']:
#                 sorted_prd = matching_results[dmd].sort_values(ascending=False)
#                 # Exclude the dmd herself
#                 most_similar_prd = dict(sorted_prd[:15])
#                 most_similar_prd_for_dmd[dmd] = most_similar_prd

#             # Create DataFrames to display the results
#             most_similar_prd_for_dmd_df = pd.merge(most_similar_prd_for_dmd_df, pd.DataFrame(
#                 most_similar_prd_for_dmd.items(), columns=['dmd', 'Most Similar prd ' + str(c)]))

#         most_similar_prd_for_dmd_df['total'] = most_similar_prd_for_dmd_df.apply(
#             lambda x: dict(
#                 sum(
#                     map(Counter, x.iloc[1:4].apply(lambda y: dict(y))), start=Counter()
#                 )
#             ), axis=1)
#         df = pd.DataFrame(most_similar_prd_for_dmd_df['total'].tolist()[0].items(), columns=[
#                           'ID', 'Values']).sort_values('Values', ascending=False).reset_index(drop=True).iloc[:item_number, :]
#         df.index += 1
#         df = pd.merge(df, tokenized_prd_df[['prd_urlIdentifier', 'prd_title',
#                                             'prd_key_words']], left_on='ID', right_on='prd_urlIdentifier').drop('prd_urlIdentifier', axis=1).rename(columns={'prd_title': 'Title', 'prd_key_words': 'keywords'})
#         df['Links'] = df['ID'].apply(
#             lambda r: f'<a href="https://techmart.ir/product/view/{r}">Link</a>')
#     # ------------------------------------------------------------------------------------------
#     # Apply the styling function to the 'Values' column
#         # apply gradient styling

#     def gradient_color(val):
#         # Convert the value to a color between red (0) and green (1)
#         color = f'rgba({int((1 - val) * 255)}, {int(val * 255)}, 0, 0.5)'
#         return [f'background-color: {color}' for _ in val]

#     styled_df = df.style.apply(gradient_color, subset=['Values'], axis=1)
#     st.write(styled_df.to_html(escape=False, index=False),
#              unsafe_allow_html=True, hide_index=True)
