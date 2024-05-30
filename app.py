# Import packages
import streamlit as st
from streamlit_tags import st_tags
import io
import XlsxWriter
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
import re
import codecs
from hazm import *
import ast
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from gensim import corpora, models
from gensim.models.coherencemodel import CoherenceModel
# -------------------------------------------------------------------------------
nmz = Normalizer()
nltk.download("punkt")
nltk.download("stopwords")
en_stops = set(stopwords.words("english"))
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

dmd_url = 'https://raw.githubusercontent.com/salehipak/military/main/tokenized_dmd_df.csv'
prd_url = 'https://raw.githubusercontent.com/salehipak/military/main/tokenized_prd_df.csv'
# -------------------------------------------------------------------------------
# title and subtitle
st.title('Technomart Matching Demo')
st.markdown("<p style='font-size:18px;'>Maching demand and supply of Iran National Technomart.</p>", unsafe_allow_html=True)
# st.text('Maching demand and supply of Iran National Technomart')
input_file = st.radio(
    "Do you have your own files?",
    options=["Yes", "No"]
    , index=1
)
if input_file == 'Yes':
# Demmender or supplier
    # st.divider()
    st.write("Please download sample file below. Your file should have :red[***.xlsx***] format and have the same columns.")
    # Create a sample DataFrame
    data = pd.DataFrame({
        'title':['دستگاه تولید بویه های صیادی']
        ,'description':['بویه های صیادی یکی از ملزومات مهم صنعت صید کشور می باشد که...']
        ,'keywords':[['ماهیگیری','صیادی']]
    })
    output = io.BytesIO()
    writer = pd.ExcelWriter(output, engine="xlsxwriter")
    data.to_excel(writer, index=False, sheet_name="sheet1")
    writer.close()
    data_bytes = output.getvalue()
    st.download_button(label="Download Sample"
                       ,mime='application/vnd.ms-excel'
                       , file_name='sample.xlsx'
                       ,data=data_bytes
                       )
    
    st.write("Now upload your demand and Supply files.") 
    upload_dmd = st.file_uploader(":red[***Demand***]")
    if upload_dmd is not None:
        uploaded_dmd_df = pd.read_excel(upload_dmd)

        tokenized_uploaded_dmd_df = tokenize(uploaded_dmd_df, ['title', 'description', 'keywords'])
        st.write(uploaded_dmd_df.head(5))
    
    upload_prd = st.file_uploader(":red[***Supply***]")
    if upload_prd is not None:
        uploaded_prd_df = pd.read_excel(upload_prd)
        tokenized_uploaded_prd_df = tokenize(uploaded_prd_df, ['title', 'description', 'keywords'])
        st.write(uploaded_prd_df.head(5))
    st.divider()

input_type = st.radio(
    "Are you supplier or demander?",
    key="visibility",
    options=["Supplier", "Demander"]
)
# Input Title, Description and Keywords
user_input_title = st.text_input(
    "Enter Title:", placeholder='Please enter your title here.', max_chars=100)
user_input_description = st.text_area(
    "Enter Description:", placeholder='Please enter your description here.', max_chars=2000, height=200)
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
                        ('Cosine Similarity', 'Jaccard Similarity', 'Cosine + LDA', 'Jaccard + LDA'))

with col2:
    item_number = st.selectbox('Number',
                               set(range(1, 10)))
st.text("")
st.text("")
# -------------------------------------------------------------------------------
# RUN Button and Result
button_id = st.button("Run", key="my_button", type='primary',
                      use_container_width=True)
if button_id:
    js_code = """
        <script>
            const button = document.querySelector('[data-baseweb="button"]');

            button.addEventListener('click', function() {
                button.style.backgroundColor = 'red';
            });
        </script>
    """
    st.markdown(js_code, unsafe_allow_html=True)
    # --------------------------------------------------------------------------------
    # Supplier
    if input_type == 'Supplier':
        prd_df = pd.DataFrame({'prd_urlIdentifier': 'PRD--1', 'prd_title': [user_input_title], 'prd_description': [user_input_description], 'prd_key_words': str([user_input_keywords])
                               })
        tokenized_prd_df = tokenize(
            prd_df, ['prd_title', 'prd_description', 'prd_key_words'])
        tokenized_dmd_df = pd.read_csv(dmd_url, converters={'tokenized_dmd_title': ast.literal_eval, 'tokenized_dmd_description': ast.literal_eval, 'tokenized_dmd_key_words': ast.literal_eval}
                                       )

        most_similar_dmd_for_prd_df = pd.DataFrame(
            {'prd': prd_df['prd_urlIdentifier']})
        for c in ['title', 'description', 'key_words']:
            # Create lists of tokenized title for dmd and prd
            dmd_token = [' '.join(tokens)
                         for tokens in tokenized_dmd_df['tokenized_dmd_' + str(c)]]
            prd_token = [' '.join(tokens)
                         for tokens in tokenized_prd_df['tokenized_prd_' + str(c)]]
        # Initialize a TF-IDF vectorizer
            tfidf_vectorizer = TfidfVectorizer()

            # Combine all title into one list
            all_token = dmd_token + prd_token

            # Fit and transform the TF-IDF vectorizer on all title
            tfidf_matrix = tfidf_vectorizer.fit_transform(all_token)

            # Calculate similarity between dmd and prd
            if algo in ['Cosine Similarity', 'Cosine + LDA']:
                similarity_matrix = cosine_similarity(
                    tfidf_matrix[:len(dmd_token)], tfidf_matrix[len(dmd_token):])
                d = {'title': 0.2, 'description': 0.35, 'key_words': 0.4}
                matching_results = pd.DataFrame(
                    similarity_matrix * d[c], index=tokenized_dmd_df['dmd_urlIdentifier'], columns=tokenized_prd_df['prd_urlIdentifier'])
            elif algo in ['Jaccard Similarity', 'Jaccard + LDA']:
                similarity_matrix = tfidf_matrix[:len(dmd_token)].dot(
                    tfidf_matrix[len(dmd_token):].T)
                d = {'title': 0.6, 'description': 0.4, 'key_words': 0.4}
                matching_results = pd.DataFrame(similarity_matrix.toarray(
                ) * d[c], index=tokenized_dmd_df['dmd_urlIdentifier'], columns=tokenized_prd_df['prd_urlIdentifier'])

            # Find the most similar dmd for each prd
            most_similar_dmd_for_prd = {}
            for prd in prd_df['prd_urlIdentifier']:
                sorted_dmd = matching_results[prd].sort_values(ascending=False)
                # Exclude the prd herself
                most_similar_dmd = dict(sorted_dmd[:100])
                most_similar_dmd_for_prd[prd] = most_similar_dmd

            # Create DataFrames to display the results
            most_similar_dmd_for_prd_df = pd.merge(most_similar_dmd_for_prd_df, pd.DataFrame(
                most_similar_dmd_for_prd.items(), columns=['prd', 'Most Similar dmd ' + str(c)]))

        most_similar_dmd_for_prd_df['total'] = most_similar_dmd_for_prd_df.apply(
            lambda x: dict(
                sum(
                    map(Counter, x.iloc[1:4].apply(lambda y: dict(y))), start=Counter()
                )
            ), axis=1)

        df = pd.DataFrame(most_similar_dmd_for_prd_df['total'].tolist()[0].items(), columns=[
                          'ID', 'Values']).sort_values('Values', ascending=False).reset_index(drop=True).iloc[:item_number, :]
        df.Values = df.Values.round(2)
        df.index += 1
        df = pd.merge(df, tokenized_dmd_df[['dmd_urlIdentifier', 'dmd_title',
                                            'dmd_key_words']], left_on='ID', right_on='dmd_urlIdentifier').drop('dmd_urlIdentifier', axis=1).rename(columns={'dmd_title': 'Title', 'dmd_key_words': 'keywords'})
        if algo in ['Cosine + LDA', 'Jaccard + LDA']:
            tokenized_documents = tokenized_dmd_df['tokenized_dmd_description'].tolist(
            ) + tokenized_prd_df['tokenized_prd_description'].tolist()
            dictionary = corpora.Dictionary(tokenized_documents)

            corpus = [dictionary.doc2bow(doc) for doc in tokenized_documents]
            # Train the LDA model
            lda_model = models.LdaModel(
                corpus, num_topics=5, id2word=dictionary, passes=5)

            # Infer topic proportions for each document
            topic_proportions = [lda_model[doc] for doc in corpus]

            # Automatically generate topic labels based on top words
            topic_labels = {}
            for topic_id in range(lda_model.num_topics):
                topic_words = [word for word,
                               prob in lda_model.show_topic(topic_id)]
                topic_labels[topic_id] = ', '.join(topic_words)

            # Label documents with topics
            document_labels = [max(topic_dist, key=lambda x: x[1])[0]
                               for topic_dist in topic_proportions]

            # Assign automatically generated topic labels to the documents
            labeled_documents = [topic_labels[label]
                                 for label in document_labels]
            tokenized_dmd_df['lda_dmd_description'] = labeled_documents[:len(
                tokenized_dmd_df)]
            df = df[df['ID'].isin(tokenized_dmd_df['dmd_urlIdentifier']
                                  [tokenized_dmd_df['lda_dmd_description'] == labeled_documents[-1]])].reset_index()
        df['Link'] = df['ID'].apply(
            lambda r: f'<a href="https://techmart.ir/demand/view/{r}">Link</a>')

    # ----------------------------------------------------------------------------------
    # Demander
    elif input_type == 'Demander':
        dmd_df = pd.DataFrame({'dmd_urlIdentifier': 'dmd--1', 'dmd_title': [user_input_title], 'dmd_description': [user_input_description], 'dmd_key_words': str([user_input_keywords])
                               })
        tokenized_dmd_df = tokenize(
            dmd_df, ['dmd_title', 'dmd_description', 'dmd_key_words'])
        tokenized_prd_df = pd.read_csv(prd_url, converters={'tokenized_prd_title': ast.literal_eval, 'tokenized_prd_description': ast.literal_eval, 'tokenized_prd_key_words': ast.literal_eval}
                                       )

        most_similar_prd_for_dmd_df = pd.DataFrame(
            {'dmd': dmd_df['dmd_urlIdentifier']})
        for c in ['title', 'description', 'key_words']:
            # Create lists of tokenized title for prd and dmd
            prd_token = [' '.join(tokens)
                         for tokens in tokenized_prd_df['tokenized_prd_' + str(c)]]
            dmd_token = [' '.join(tokens)
                         for tokens in tokenized_dmd_df['tokenized_dmd_' + str(c)]]
        # Initialize a TF-IDF vectorizer
            tfidf_vectorizer = TfidfVectorizer()

            # Combine all title into one list
            all_token = prd_token + dmd_token

            # Fit and transform the TF-IDF vectorizer on all title
            tfidf_matrix = tfidf_vectorizer.fit_transform(all_token)

            # Calculate similarity between prd and dmd
            if algo in ['Cosine Similarity', 'Cosine + LDA']:
                similarity_matrix = cosine_similarity(
                    tfidf_matrix[:len(prd_token)], tfidf_matrix[len(prd_token):])
                d = {'title': 0.2, 'description': 0.35, 'key_words': 0.4}
                matching_results = pd.DataFrame(
                    similarity_matrix * d[c], index=tokenized_prd_df['prd_urlIdentifier'], columns=tokenized_dmd_df['dmd_urlIdentifier'])
            elif algo in ['Jaccard Similarity', 'Jaccard + LDA']:
                similarity_matrix = tfidf_matrix[:len(prd_token)].dot(
                    tfidf_matrix[len(prd_token):].T)
                d = {'title': 0.6, 'description': 0.4, 'key_words': 0.4}
                matching_results = pd.DataFrame(similarity_matrix.toarray(
                ) * d[c], index=tokenized_prd_df['prd_urlIdentifier'], columns=tokenized_dmd_df['dmd_urlIdentifier'])

            # Find the most similar prd for each dmd
            most_similar_prd_for_dmd = {}
            for dmd in dmd_df['dmd_urlIdentifier']:
                sorted_prd = matching_results[dmd].sort_values(ascending=False)
                # Exclude the dmd herself
                most_similar_prd = dict(sorted_prd[:100])
                most_similar_prd_for_dmd[dmd] = most_similar_prd

            # Create DataFrames to display the results
            most_similar_prd_for_dmd_df = pd.merge(most_similar_prd_for_dmd_df, pd.DataFrame(
                most_similar_prd_for_dmd.items(), columns=['dmd', 'Most Similar prd ' + str(c)]))

        most_similar_prd_for_dmd_df['total'] = most_similar_prd_for_dmd_df.apply(
            lambda x: dict(
                sum(
                    map(Counter, x.iloc[1:4].apply(lambda y: dict(y))), start=Counter()
                )
            ), axis=1)
        df = pd.DataFrame(most_similar_prd_for_dmd_df['total'].tolist()[0].items(), columns=[
                          'ID', 'Values']).sort_values('Values', ascending=False).reset_index(drop=True).iloc[:item_number, :]
        df.Values = df.Values.round(2)
        df.index += 1
        df = pd.merge(df, tokenized_prd_df[['prd_urlIdentifier', 'prd_title',
                                            'prd_key_words']], left_on='ID', right_on='prd_urlIdentifier').drop('prd_urlIdentifier', axis=1).rename(columns={'prd_title': 'Title', 'prd_key_words': 'keywords'})
        if algo in ['Cosine + LDA', 'Jaccard + LDA']:
            tokenized_documents = tokenized_prd_df['tokenized_prd_description'].tolist(
            ) + tokenized_dmd_df['tokenized_dmd_description'].tolist()
            dictionary = corpora.Dictionary(tokenized_documents)

            corpus = [dictionary.doc2bow(doc) for doc in tokenized_documents]
            # Train the LDA model
            lda_model = models.LdaModel(
                corpus, num_topics=5, id2word=dictionary, passes=5)

            # Infer topic proportions for each document
            topic_proportions = [lda_model[doc] for doc in corpus]

            # Automatically generate topic labels based on top words
            topic_labels = {}
            for topic_id in range(lda_model.num_topics):
                topic_words = [word for word,
                               prob in lda_model.show_topic(topic_id)]
                topic_labels[topic_id] = ', '.join(topic_words)

            # Label documents with topics
            document_labels = [max(topic_dist, key=lambda x: x[1])[0]
                               for topic_dist in topic_proportions]

            # Assign automatically generated topic labels to the documents
            labeled_documents = [topic_labels[label]
                                 for label in document_labels]
            tokenized_prd_df['lda_prd_description'] = labeled_documents[:len(
                tokenized_prd_df)]

            df = df[df['ID'].isin(tokenized_prd_df['prd_urlIdentifier']
                                  [tokenized_prd_df['lda_prd_description'] == labeled_documents[-1]])].reset_index()
        
        df['Link'] = df['ID'].apply(
            lambda r: f'<a href="https://techmart.ir/product/view/{r}">Link</a>')
    # ------------------------------------------------------------------------------------------
    # Apply the styling function to the 'Values' column
        # apply gradient styling

    def gradient_color(val):
        # Convert the value to a color between red (0) and green (1)
        color = f'rgba({int((1 - val) * 255)}, {int(val * 255)}, 0, 0.5)'
        return [f'background-color: {color}' for _ in val]

    styled_df = df.style.apply(gradient_color, subset=['Values'], axis=1)
    st.write(styled_df.to_html(escape=False, index=False),
             unsafe_allow_html=True, hide_index=True)
