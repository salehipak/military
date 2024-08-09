# Import packages
import streamlit as st
from streamlit_tags import st_tags
import io
import xlsxwriter
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
from sklearn.metrics import jaccard_score
from gensim import corpora, models
from gensim.models.coherencemodel import CoherenceModel

def jaccard_similarity(set1, set2):
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union
    
def gradient_color(val):
    # Convert the value to a color between red (0) and green (1)
    color = f'rgba({int((1 - val) * 255)}, {int(val * 255)}, 0, 0.5)'
    return [f'background-color: {color}' for _ in val]
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

input_type = st.radio(
    "Are you supplier or demander?",
    key="visibility",
    options=["Supplier", "Demander"]
)
input_file = st.radio(
    "Do you want to upload a file?",
    options=["Yes", "No"]
    , index=1
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
                        ('Cosine Similarity', 'Jaccard Similarity', 'LDA'))

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
        if input_file == 'Yes':
        # Demmender or supplier
            # st.divider()
            st.write("Please download sample file below. Your file should have :red[***.xlsx***] format with the same columns.")
            # Create a sample DataFrame
            data = pd.DataFrame({
                'id':[1]
                , 'title':['دستگاه تولید بویه های صیادی']
                ,'description':['بویه های صیادی یکی از ملزومات مهم صنعت صید کشور می باشد که...']
                ,'key_words':[['ماهیگیری','صیادی']]
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
            
            st.write("Now upload your Supply file.")
            upload_prd = st.file_uploader(":red[***Supply***]")
            if upload_prd is not None:
                uploaded_prd_df = pd.read_excel(upload_prd).sort_values('id')
                uploaded_prd_df.insert(loc=2,column='Identifier',value= ['Manual_PRD_' + str(_ + 1) for _ in range(len(uploaded_prd_df))])
                st.write(uploaded_prd_df.head(5))
                prd_df = upload_prd.rename(columns = {'id':'prd_id','title':'prd_title','urlIdentifier':'prd_urlIdentifier','description':'prd_description','key_words':'prd_key_words'})
            else:
                prd_df = pd.DataFrame({'prd_urlIdentifier': 'PRD--1', 'prd_title': [user_input_title], 'prd_description': [user_input_description], 'prd_key_words': str([user_input_keywords])
                               })
                
        tokenized_prd_df = tokenize(
            prd_df, ['prd_title', 'prd_description', 'prd_key_words'])
        tokenized_dmd_df = pd.read_csv('tokenized_dmd_df.csv', converters={'tokenized_dmd_title': ast.literal_eval, 'tokenized_dmd_description': ast.literal_eval, 'tokenized_dmd_key_words': ast.literal_eval}
                                       )
        st.divider()
        
        if algo == 'LDA':
          tokenized_documents = (tokenized_dmd_df['tokenized_dmd_title'] + tokenized_dmd_df['tokenized_dmd_description'] + tokenized_dmd_df['tokenized_dmd_key_words']).tolist() + (tokenized_prd_df['tokenized_prd_title'] + tokenized_prd_df['tokenized_prd_description'] + tokenized_prd_df['tokenized_prd_key_words']).tolist() 
          dictionary = corpora.Dictionary(tokenized_documents)
          corpus = [dictionary.doc2bow(doc) for doc in tokenized_documents]
          lda_model = models.LdaModel(corpus,random_state=1234, num_topics=4, id2word=dictionary, passes=10)
          topic_proportions = [lda_model[doc] for doc in corpus]
          
          topic_labels = {}
          for topic_id in range(lda_model.num_topics):
              topic_words = [word for word, prob in lda_model.show_topic(topic_id)]
              topic_labels[topic_id] = ', '.join(topic_words)

          document_labels = [max(topic_dist, key=lambda x: x[1])[0] for topic_dist in topic_proportions]
          labeled_documents = [topic_labels[label] for label in document_labels]
          tokenized_dmd_df['lda_dmd'] = labeled_documents[:len(
                tokenized_dmd_df)]
            
          df = tokenized_dmd_df[tokenized_dmd_df['lda_dmd'] == labeled_documents[-1]]
          df = df[['dmd_urlIdentifier', 'dmd_title','dmd_key_words','lda_dmd']].rename(columns={'dmd_title': 'Title', 'dmd_key_words': 'keywords','dmd_urlIdentifier':'ID','lda_dmd':'Label'}).iloc[-item_number:, :].reset_index()
          df.index += 1
          df['Link'] = np.where(df['ID'].str.contains('Manual'),'-',df['ID'].apply(
          lambda r: f'<a href="https://techmart.ir/demand/view/{r}">Link</a>'))
          st.write(df.to_html(escape=False, index=False),
                 unsafe_allow_html=True, hide_index=True)
        
        else:
            if algo == 'Cosine Similarity':
                most_similar_dmd_for_prd_df = pd.DataFrame({'prd': prd_df['prd_urlIdentifier']})
                for c in ['title', 'description', 'key_words']:
                    dmd_token = [' '.join(tokens)
                                 for tokens in tokenized_dmd_df['tokenized_dmd_' + str(c)]]
                    prd_token = [' '.join(tokens)
                                 for tokens in tokenized_prd_df['tokenized_prd_' + str(c)]]
                    tfidf_vectorizer = TfidfVectorizer()
                    
                    all_token = dmd_token + prd_token
                    tfidf_matrix = tfidf_vectorizer.fit_transform(all_token)
                    similarity_matrix = cosine_similarity(tfidf_matrix[:len(dmd_token)], tfidf_matrix[len(dmd_token):])
                    matching_results = pd.DataFrame(similarity_matrix, index=tokenized_dmd_df['dmd_urlIdentifier'], columns=tokenized_prd_df['prd_urlIdentifier'])
                    
                    most_similar_dmd_for_prd = {}
                    for prd in prd_df['prd_urlIdentifier']:
                        sorted_dmd = matching_results[prd].sort_values(ascending=False)
                        # Exclude the prd herself
                        most_similar_dmd = dict(sorted_dmd[:100])
                        most_similar_dmd_for_prd[prd] = most_similar_dmd
    
                    most_similar_dmd_for_prd_df = pd.merge(most_similar_dmd_for_prd_df, pd.DataFrame(
                    most_similar_dmd_for_prd.items(), columns=['prd', 'Most Similar dmd ' + str(c)]))
                    
                most_similar_dmd_for_prd_df['total'] = most_similar_dmd_for_prd_df.apply(lambda x: dict(sum(map(Counter, x.iloc[1:4].apply(lambda y: dict(y))), start=Counter())), axis=1)
    
            elif algo == 'Jaccard Similarity':    
                most_similar_dmd_for_prd_df = pd.DataFrame({'prd': prd_df['prd_urlIdentifier']})
                for c in ['title', 'description', 'key_words']:
                    dmd_token = [set(tokens) for tokens in tokenized_dmd_df['tokenized_dmd_' + str(c)]]
                    prd_token = [set(tokens) for tokens in tokenized_prd_df['tokenized_prd_' + str(c)]]
                    
                    similarity_matrix = np.zeros((len(dmd_token), len(prd_token)))
                    for i, dmd_tokens in enumerate(dmd_token):
                        for j, prd_tokens in enumerate(prd_token):
                            similarity_matrix[i, j] = jaccard_similarity(dmd_tokens, prd_tokens)
                    matching_results = pd.DataFrame(similarity_matrix, index=tokenized_dmd_df['dmd_urlIdentifier'], columns=tokenized_prd_df['prd_urlIdentifier'])
                    most_similar_dmd_for_prd = {}
                    for prd in prd_df['prd_urlIdentifier']:
                        sorted_dmd = matching_results[prd].sort_values(ascending=False)
                        # Exclude the prd herself
                        most_similar_dmd = dict(sorted_dmd[:100])
                        most_similar_dmd_for_prd[prd] = most_similar_dmd
    
                    most_similar_dmd_for_prd_df = pd.merge(most_similar_dmd_for_prd_df, pd.DataFrame(
                    most_similar_dmd_for_prd.items(), columns=['prd', 'Most Similar dmd ' + str(c)]))
                    
                most_similar_dmd_for_prd_df['total'] = most_similar_dmd_for_prd_df.apply(lambda x: dict(sum(map(Counter, x.iloc[1:4].apply(lambda y: dict(y))), start=Counter())), axis=1)
              
            df = pd.DataFrame(most_similar_dmd_for_prd_df['total'].tolist()[0].items(), columns=[
                          'ID', 'Values']).sort_values('Values', ascending=False).reset_index(drop=True).iloc[:item_number, :]
            df.Values = df.Values.round(2)
            df.index += 1
            df = pd.merge(df, tokenized_dmd_df[['dmd_urlIdentifier', 'dmd_title',
                                            'dmd_key_words']], left_on='ID', right_on='dmd_urlIdentifier').drop('dmd_urlIdentifier', axis=1).rename(columns={'dmd_title': 'Title', 'dmd_key_words': 'keywords'})
        
            df['Link'] = np.where(df['ID'].str.contains('Manual'),'-',df['ID'].apply(lambda r: f'<a href="https://techmart.ir/demand/view/{r}">Link</a>'))

            styled_df = df.style.apply(gradient_color, subset=['Values'], axis=1)
            st.write(styled_df.to_html(escape=False, index=False),unsafe_allow_html=True, hide_index=True)
#----------------------------------
    if input_file == 'Yes':
        if input_file == 'Yes':
        # Demmender or supplier
            # st.divider()
            st.write("Please download sample file below. Your file should have :red[***.xlsx***] format with the same columns.")
            # Create a sample DataFrame
            data = pd.DataFrame({
                'id':[1]
                , 'title':['دستگاه تولید بویه های صیادی']
                ,'description':['بویه های صیادی یکی از ملزومات مهم صنعت صید کشور می باشد که...']
                ,'key_words':[['ماهیگیری','صیادی']]
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
            
            st.write("Now upload your demand file.") 
            upload_dmd = st.file_uploader(":red[***Demand***]")
            if upload_dmd is not None:
                uploaded_dmd_df = pd.read_excel(upload_dmd).sort_values('id')
                uploaded_dmd_df.insert(loc=2,column='Identifier',value= ['Manual_DMD_' + str(_ + 1) for _ in range(len(uploaded_dmd_df))])
                st.write(uploaded_dmd_df.head(5))
            else:
                dmd_df = pd.DataFrame({'dmd_urlIdentifier': 'DMD--1', 'dmd_title': [user_input_title], 'dmd_description': [user_input_description], 'dmd_key_words': str([user_input_keywords])
                               })
                
        tokenized_dmd_df = tokenize(
            dmd_df, ['dmd_title', 'dmd_description', 'dmd_key_words'])
        tokenized_dmd_df = pd.read_csv('tokenized_dmd_df.csv', converters={'tokenized_dmd_title': ast.literal_eval, 'tokenized_dmd_description': ast.literal_eval, 'tokenized_dmd_key_words': ast.literal_eval}
                                       )
        st.divider()
        

        if algo == 'LDA':
          tokenized_documents = (tokenized_prd_df['tokenized_prd_title'] + tokenized_prd_df['tokenized_prd_description'] + tokenized_prd_df['tokenized_prd_key_words']).tolist() + (tokenized_dmd_df['tokenized_dmd_title'] + tokenized_dmd_df['tokenized_dmd_description'] + tokenized_dmd_df['tokenized_dmd_key_words']).tolist() 
          dictionary = corpora.Dictionary(tokenized_documents)
          corpus = [dictionary.doc2bow(doc) for doc in tokenized_documents]
          lda_model = models.LdaModel(corpus,random_state=1234, num_topics=4, id2word=dictionary, passes=10)
          topic_proportions = [lda_model[doc] for doc in corpus]
          
          topic_labels = {}
          for topic_id in range(lda_model.num_topics):
              topic_words = [word for word, prob in lda_model.show_topic(topic_id)]
              topic_labels[topic_id] = ', '.join(topic_words)

          document_labels = [max(topic_dist, key=lambda x: x[1])[0] for topic_dist in topic_proportions]
          labeled_documents = [topic_labels[label] for label in document_labels]
          tokenized_prd_df['lda_prd'] = labeled_documents[:len(
                tokenized_prd_df)]
            
          df = tokenized_prd_df[tokenized_prd_df['lda_prd'] == labeled_documents[-1]]
          df = df[['prd_urlIdentifier', 'prd_title','prd_key_words','lda_prd']].rename(columns={'prd_title': 'Title', 'prd_key_words': 'keywords','prd_urlIdentifier':'ID','lda_prd':'Label'}).iloc[-item_number:, :].reset_index()
          df.index += 1
          df['Link'] = np.where(df['ID'].str.contains('Manual'),'-',df['ID'].apply(
          lambda r: f'<a href="https://techmart.ir/demand/view/{r}">Link</a>'))
          st.write(df.to_html(escape=False, index=False),
                 unsafe_allow_html=True, hide_index=True)
        
        else:
            if algo == 'Cosine Similarity':
                most_similar_prd_for_dmd_df = pd.DataFrame({'dmd': dmd_df['dmd_urlIdentifier']})
                for c in ['title', 'description', 'key_words']:
                    prd_token = [' '.join(tokens)
                                 for tokens in tokenized_prd_df['tokenized_prd_' + str(c)]]
                    dmd_token = [' '.join(tokens)
                                 for tokens in tokenized_dmd_df['tokenized_dmd_' + str(c)]]
                    tfidf_vectorizer = TfidfVectorizer()
                    
                    all_token = prd_token + dmd_token
                    tfidf_matrix = tfidf_vectorizer.fit_transform(all_token)
                    similarity_matrix = cosine_similarity(tfidf_matrix[:len(prd_token)], tfidf_matrix[len(prd_token):])
                    matching_results = pd.DataFrame(similarity_matrix, index=tokenized_prd_df['prd_urlIdentifier'], columns=tokenized_dmd_df['dmd_urlIdentifier'])
                    
                    most_similar_prd_for_dmd = {}
                    for dmd in dmd_df['dmd_urlIdentifier']:
                        sorted_prd = matching_results[dmd].sort_values(ascending=False)
                        # Exclude the dmd herself
                        most_similar_prd = dict(sorted_prd[:100])
                        most_similar_prd_for_dmd[dmd] = most_similar_prd
    
                    most_similar_prd_for_dmd_df = pd.merge(most_similar_prd_for_dmd_df, pd.DataFrame(
                    most_similar_prd_for_dmd.items(), columns=['dmd', 'Most Similar prd ' + str(c)]))
                    
                most_similar_prd_for_dmd_df['total'] = most_similar_prd_for_dmd_df.apply(lambda x: dict(sum(map(Counter, x.iloc[1:4].apply(lambda y: dict(y))), start=Counter())), axis=1)
    
            elif algo == 'Jaccard Similarity':    
                most_similar_prd_for_dmd_df = pd.DataFrame({'dmd': dmd_df['dmd_urlIdentifier']})
                for c in ['title', 'description', 'key_words']:
                    prd_token = [set(tokens) for tokens in tokenized_prd_df['tokenized_prd_' + str(c)]]
                    dmd_token = [set(tokens) for tokens in tokenized_dmd_df['tokenized_dmd_' + str(c)]]
                    
                    similarity_matrix = np.zeros((len(prd_token), len(dmd_token)))
                    for i, prd_tokens in enumerate(prd_token):
                        for j, dmd_tokens in enumerate(dmd_token):
                            similarity_matrix[i, j] = jaccard_similarity(prd_tokens, dmd_tokens)
                    matching_results = pd.DataFrame(similarity_matrix, index=tokenized_prd_df['prd_urlIdentifier'], columns=tokenized_dmd_df['dmd_urlIdentifier'])
                    most_similar_prd_for_dmd = {}
                    for dmd in dmd_df['dmd_urlIdentifier']:
                        sorted_prd = matching_results[dmd].sort_values(ascending=False)
                        # Exclude the dmd herself
                        most_similar_prd = dict(sorted_prd[:100])
                        most_similar_prd_for_dmd[dmd] = most_similar_prd
    
                    most_similar_prd_for_dmd_df = pd.merge(most_similar_prd_for_dmd_df, pd.DataFrame(
                    most_similar_prd_for_dmd.items(), columns=['dmd', 'Most Similar prd ' + str(c)]))
                    
                most_similar_prd_for_dmd_df['total'] = most_similar_prd_for_dmd_df.apply(lambda x: dict(sum(map(Counter, x.iloc[1:4].apply(lambda y: dict(y))), start=Counter())), axis=1)
              
            df = pd.DataFrame(most_similar_prd_for_dmd_df['total'].tolist()[0].items(), columns=[
                          'ID', 'Values']).sort_values('Values', ascending=False).reset_index(drop=True).iloc[:item_number, :]
            df.Values = df.Values.round(2)
            df.index += 1
            df = pd.merge(df, tokenized_prd_df[['prd_urlIdentifier', 'prd_title',
                                            'prd_key_words']], left_on='ID', right_on='prd_urlIdentifier').drop('prd_urlIdentifier', axis=1).rename(columns={'prd_title': 'Title', 'prd_key_words': 'keywords'})
        
            df['Link'] = np.where(df['ID'].str.contains('Manual'),'-',df['ID'].apply(lambda r: f'<a href="https://techmart.ir/demand/view/{r}">Link</a>'))

            styled_df = df.style.apply(gradient_color, subset=['Values'], axis=1)
            st.write(styled_df.to_html(escape=False, index=False),unsafe_allow_html=True, hide_index=True)

    # ------------------------------------------------------------------------------------------

