##### Natural Language Processing Application (Streamlit) ####

## Importing the libraries

import json
from collections import Counter
from functools import partial
from random import random
from warnings import simplefilter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import requests
import spacy
import streamlit as st
import textacy
import textacy.tm
from keybert import KeyBERT
from keyphrase_vectorizers import KeyphraseCountVectorizer
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from spacy import displacy
from streamlit_option_menu import option_menu
from wordcloud import WordCloud
from umap import UMAP
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from cluestar import plot_text

simplefilter(action='ignore', category=FutureWarning)

# Configuring the main page, defining functions that will be used across multiple tasks,//
# and setting the Session State objects (this is data that is kept in memory while the //
# user navigates around through the various tasks).

st.set_page_config(page_title="Natural Language Processing Tool", page_icon="🤖")

st.title("Natural Language Processing Tool")

with st.sidebar:
    selected = option_menu("Method Menu", ['Home', 'Basic Text Analysis', 'Named Entity Recognition', 'Text Categorization', 'Text Summarization','Topic Modeling','Document Clustering'], icons=['house'], menu_icon="cast", default_index=0)

def plot_result(top_topics, scores):
    top_topics = np.array(top_topics)
    scores = np.array(scores)
    scores *= 100
    fig = px.bar(x=scores, y=top_topics, orientation='h', 
                 labels={'x': 'Probability', 'y': 'Category'},
                 text=scores,
                 range_x=(0,115),
                 title='Top Predictions',
                 color=np.linspace(0,1,len(scores)),
                 color_continuous_scale="Bluered")
    fig.update(layout_coloraxis_showscale=False)
    fig.update_traces(texttemplate='%{text:0.1f}%', textposition='outside')
    st.plotly_chart(fig) 

#### While some tasks (Basic Text Analysis, Named Entity Recognition, and Topic Modeling) //
# are run locally, the others (Text Categorization, Text Summarization, and Document Clustering) //
# require making API calls to models in the Hugging Face model hub. The following 7 lines of code //
# are used to set up the API calls to the model hub.

headers = {'Authorization': st.secrets['api_key']}

def query(payload):
    data = json.dumps(payload)
    response = requests.request("POST", API_URL, headers=headers, data=data)
    return json.loads(response.content.decode("utf-8"))

if "text" not in st.session_state:
    st.session_state.text = ''

if "multitext_all" not in st.session_state:
    st.session_state.multitext_all = ''

if "multitext_short" not in st.session_state:
    st.session_state.multitext_short = ''

## The sidebar navigation pane you see on the app is accessed through "if statements". //
# Below you will see each task tab corresponds to an "if statement", and that some of the //
# tasks have sub-"if statements" nested within them.

if selected == 'Home':
    
    st.write("_**Introduction:** This web application allows you to use machine learning to analyze text. On the left-hand side of the screen you can see six analysis tasks that can be applied to your text._")

    st.write("**Instructions:** The first step is to choose how you would like to input your text. You can either type it into the text box (single document), or upload a spreadsheet with multiple rows of text (multiple documents). Once you have saved and submitted the text, you can proceed to the analysis tasks.")  
    
    choice = st.radio('Please select one of the following options:', ['Type in Text Box', 'Upload Spreadsheet with Text Column'])
    if choice == 'Type in Text Box':
        text = st.text_area('Enter Text Below:', height=200)
        submit1 = st.button('Save and Submit Text')
        if submit1:    
            if len(text) < 1:
                st.write("Please enter text above.")
            else:
                st.session_state.text = text
                st.session_state.multitext_all = ''
                st.session_state.multitext_short = ''
                st.write('Text saved successfully! You are now ready to try out the analysis tasks.')
        
    if choice == 'Upload Spreadsheet with Text Column':
        st.write("Below is an example of a properly formatted spreadsheet file. The first column should contain the text you would like analyzed. Additional columns, such as 'Source' and 'Date' in the example below, are optional.")
        demo = {'Text': ['This is the first sentence.', 'And this is the second', 'And this is the third.'], 'Source': ['Internet', 'Newspaper', 'TV'], 'Date': ['2/13/2021','4/27/2021','8/1/2021']}
        demo1 = pd.DataFrame(demo)
        st.dataframe(demo1)
        dfs = st.file_uploader("Please drag and drop your file into the space below (.csv files only)", type=["csv"])
        submit2 = st.button('Save and Submit Text')
        if submit2:
            if dfs is None:
                st.write("Please upload a file.")
            else:
                st.write('Text uploaded successfully! You are now ready to try out the analysis tasks.')
                df = pd.read_csv(dfs, encoding_errors = 'ignore')
                first_column_all = df.iloc[1:, 0]
                first_column_short = df.iloc[1:11, 0]  
                texts_all = first_column_all.to_list()
                texts_short = first_column_short.to_list()
                test_all = [str(x) for x in texts_all]
                test_short = [str(x) for x in texts_short]
                st.session_state.multitext_all = test_all
                st.session_state.multitext_short = test_short
                st.session_state.text = ''

if selected == 'Basic Text Analysis':
    
    st.subheader("Basic Text Analysis")
    st.write("**Description:** This task involves producing simple descriptive statistics about your text.")
    submit3 = st.button('Analyze Text')
    if submit3:
        if len(st.session_state.text) > 0:
            word_list = st.session_state.text.split()
            doc_count = 1
            col1, col2 = st.columns(2)
            word_freq = Counter(word_list)
            word_freq_df = pd.DataFrame(word_freq.most_common(), columns=['Word', 'Count'])
            word_cloud = WordCloud(background_color='white', max_words=100, max_font_size=50, random_state=42).generate(' '.join(word_list))
            with col1:
                st.write("Document Count: ", doc_count)
                st.write("Unique Word Count: ", len(set(word_list)))

            with col2:
                st.write("Word Count: ", len(word_list))
                avg_word = np.mean([len(word) for word in word_list])
                st.write("Average Word Length: ", round(avg_word, 2))

            st.write('--------------------------------------------')

            st.write("Word Cloud: ")
            st.image(word_cloud.to_image())

            st.write('--------------------------------------------')            
            st.write("**Uploaded Text:** ", st.session_state.text)
                
        elif len(st.session_state.multitext_all) > 0:
            doc_count1 = len(st.session_state.multitext_all)
            word_list1 =[]
            for i in st.session_state.multitext_all:
                word_list1.extend(i.split())
            col1, col2 = st.columns(2)
            word_freq1 = Counter(word_list1)
            word_freq_df1 = pd.DataFrame(word_freq1.most_common(), columns=['Word', 'Count'])
            word_cloud1 = WordCloud(background_color='white', max_words=100, max_font_size=50, random_state=42).generate(' '.join(word_list1))
            with col1:
                st.write("Document Count: ", doc_count1)
                st.write("Unique Word Count: ", len(set(word_list1)))
            with col2:
                st.write("Word Count: ", len(word_list1))
                avg_word1 = np.mean([len(word) for word in word_list1])
                st.write("Average Word Length: ", round(avg_word1, 2))
            
            st.write('--------------------------------------------')

            pipe = make_pipeline(TfidfVectorizer(), UMAP(random_state=42))
            X = pipe.fit_transform(st.session_state.multitext_all)
            temp_chart = plot_text(X, st.session_state.multitext_all)
            st.altair_chart(temp_chart, use_container_width=True)

            st.write('--------------------------------------------')

            st.write("Word Cloud: ")
            st.image(word_cloud1.to_image())

            st.write('--------------------------------------------')

            df = pd.DataFrame(st.session_state.multitext_all, columns=['Documents'])
            st.write("**Uploaded Text Below:** ")
            st.table(df)

        else:
            st.write("Please enter text or upload file on the Home page.")    

if selected == 'Named Entity Recognition':
    st.subheader("Named Entity Recognition")
    st.write("**Description:** This task involves identifying geopolitical entities, organizations, people, and locations in a body of text.")
    submit4 = st.button('Analyze Text')
    nlp = spacy.load('en_core_web_sm')

    if submit4:
        if len(st.session_state.text) > 0:
            entities = []
            entityLabels = []
            doc_NER = nlp(st.session_state.text)
            if len(doc_NER.ents) > 0:
                for i in doc_NER.ents:
                    entities.append(i.text)
                    entityLabels.append(i.label_)
                df14 = pd.DataFrame({'Entity': entities, 'Type of Entity': entityLabels})
                options1 = ["ORG", "GPE", "PERSON", "LOC"]
                df14 = df14[df14['Type of Entity'].isin(options1)]
                df14['Type of Entity'] = df14['Type of Entity'].str.replace('PERSON', 'Person').str.replace('ORG', 'Organization').str.replace('LOC', 'Location').str.replace('GPE', 'Geopolitical Entity')
                df14.sort_values(['Type of Entity'], ascending=False, inplace=True)
                df14 = df14.groupby(["Type of Entity", "Entity"]).size().reset_index(name="Count").sort_values(by=["Count","Type of Entity"], ascending=False)
		
                st.write(df14)
                NER_Results1 = df14.to_csv(index=False, header=True)
                st.download_button("Download Results", NER_Results1, file_name="NER_Results.csv")
                options = {"ents": ["ORG", "GPE","PERSON", "LOC"]}
                HTML_WRAPPER = """<div style="overflow-x: auto; border: 1px solid #e6e9ef; border-radius: 0.25rem; padding: 1rem; margin-bottom: 2.5rem">{}</div>"""
                html = displacy.render(doc_NER, style="ent", options=options)
                html = html.replace("\n", " ")
                st.write('--------------------------------------------')
                st.write(HTML_WRAPPER.format(html), unsafe_allow_html=True)
            else:
                st.write("No named entities found in the text.")

        elif len(st.session_state.multitext_all) > 0:
            entities1 = []
            entityLabels1 = []
            for i in st.session_state.multitext_all:
                doc_NER1 = nlp(i)
                if len(doc_NER1.ents) > 0:
                    for j in doc_NER1.ents:
                        entities1.append(j.text)
                        entityLabels1.append(j.label_)
            df3 = pd.DataFrame({'Entity': entities1, 'Type of Entity': entityLabels1})
            options1 = ["ORG", "GPE", "PERSON", "LOC"]
            df3 = df3[df3['Type of Entity'].isin(options1)]
            df3['Type of Entity'] = df3['Type of Entity'].str.replace('PERSON', 'Person').str.replace('ORG', 'Organization').str.replace('LOC', 'Location').str.replace('GPE', 'Geopolitical Entity')
            df3.sort_values(['Type of Entity'], ascending=False, inplace=True)
            df3 = df3.groupby(["Type of Entity", "Entity"]).size().reset_index(name="Count").sort_values(by=["Count","Type of Entity"], ascending=False)
            
            st.write(df3)
            NER_Results = df3.to_csv(index=False, header=True)
            st.download_button("Download Results", NER_Results, file_name="NER_Results.csv")
            
        else:
            st.write("Please enter text or upload file on the Home page.")

if selected == 'Text Categorization':
    API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-mnli"
    st.subheader("Text Categorization")
    st.write("**Description:** This task involves placing a piece of text into one or more categories.")
    
    cat1 = st.text_input('Enter each possible category name (separated by a comma). Maximum 5 categories.')
    labels1 = cat1.strip().split(',')
    submit3 = st.button('Analyze Text')
    if submit3:
        if len(st.session_state.text) > 0:
            text_class = query({"inputs": st.session_state.text, "parameters": {"candidate_labels": labels1}})
            cat1name = text_class['labels'][0]
            cat1prob = text_class['scores'][0]
            st.write('--------------------------------------------')
            st.write('**Text:** ', st.session_state.text)
            st.write('**Category:** {} | **Probability:** {:.1f}%'.format(cat1name.title(),(cat1prob*100)))
            plot_result(text_class['labels'][::-1][-10:], text_class['scores'][::-1][-10:])

        elif len(st.session_state.multitext_all) > 0:
            st.write('--------------------------------------------')
            st.write("_Note: During the current testing phase, this task can only be performed on the first 10 documents._")
            text_class1 = []
            class_name = []
            score_name = []
            my_bar = st.progress(0)
            for i in st.session_state.multitext_short:
                text_class2 = query({"inputs": i, "parameters": {"candidate_labels": labels1, "wait_for_model": True}})
                text_class1.append(i)
                class_name.append(text_class2['labels'][0])
                score_name.append(text_class2['scores'][0])
                my_bar.progress(len(class_name)*10)
            df15 = pd.DataFrame({'Text': text_class1, 'Category': class_name, 'Probability': score_name})
            st.write(df15)
            Classification_Results = df15.to_csv(index=False, header=True)
            st.download_button("Download Results", Classification_Results, file_name="Categorization_Results.csv")

        else:
            st.write("Please enter text or upload file on the Home page.")

if selected == 'Text Summarization':
    
    st.subheader("Text Summarization")
    st.write("**Description:** This task involves condensing larger bodies of text into smaller bodies of text.")
    API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-cnn"
    max_lengthy = st.slider('Maximum summary length (words)', min_value=30, max_value=120, value=50, step=5)
    min_lengthy = st.slider('Minimum summary length (words)', min_value=10, max_value=60, value=30, step=5)
    sum_choice = st.radio('If analyzing multiple texts (i.e. a CSV file), please choose how you would like to summarize your data.', ['All documents to one summary.', 'Each document summarized individually.'])
    submit5 = st.button('Analyze Text')  
    if submit5:
        if len(st.session_state.text) > 0:
            data = query({"inputs": st.session_state.text,
            "parameters": {"do_sample": False, "max_length": max_lengthy, "min_length": min_lengthy}})
            st.write('--------------------------------------------')
            st.write("**Summary:**  ", data[0]['summary_text'])
            st.write("**Original Text:**  ", st.session_state.text)

        elif len(st.session_state.multitext_all) > 0:
            st.write("_Note: During the current testing phase, this task can only be performed on the first 10 documents._")
            if sum_choice == 'All documents to one summary.':
                sum_text = ' '.join(st.session_state.multitext_short)
                sum_text2 = sum_text[0:1023]
                data = query({"inputs": sum_text2,
                "parameters": {"do_sample": False, "max_length": max_lengthy, "min_length": min_lengthy}})
                st.write('--------------------------------------------')
                st.write("**Summary:**  ", data[0]['summary_text'])
            if sum_choice == 'Each document summarized individually.':
                sum_list = []
                my_bar1 = st.progress(0)
                for i in st.session_state.multitext_short:
                    data = query({"inputs": i,
                    "parameters": {"do_sample": False, "max_length": max_lengthy, "min_length": min_lengthy}})
                    sum_list.append(data[0]['summary_text'])
                    my_bar1.progress(len(sum_list)*10)
                sum_all = list(zip(st.session_state.multitext_short, sum_list))
                sum_df = pd.DataFrame(sum_all, columns=['Text', 'Summary'])
                st.write('--------------------------------------------')
                st.write(sum_df)
                Summary_Results = sum_df.to_csv(index=False, header=True)
                st.download_button("Download Results", Summary_Results, file_name="Summary_Results.csv")
        else:
            st.write("Please enter text or upload file on the Home page.")

if selected == 'Topic Modeling':
    
    st.subheader("Topic Modeling")
    st.write("**Description:** This task involves finding topics among a collection of documents.")
    st.write("_Note: This task only applies to multiple document files (i.e. an uploaded CSV file)._")
    graph_topic_to_highlight = [0,2,4]

    num_topics = st.slider('Number of Topics to Find ', min_value=1, max_value=12, value=6, step=1)
    words_in_topic = 10
    n_grams = st.radio('N-Gram Size', ['Single Word', 'Two Word Phrases', 'Three Word Phrases'])
    model_type = st.radio('Select Topic Modeling Algorithm', ['Latent Dirichlet Allocation','Non-Negative Matrix Factorization', 'Latent Semantic Analysis'])
    if model_type == 'Non-Negative Matrix Factorization':
        model_type = 'nmf'
    elif model_type == 'Latent Dirichlet Allocation':
        model_type = 'lda'
    elif model_type == 'Latent Semantic Analysis':
        model_type = 'lsa'
    if n_grams == 'Single Word':
        n_grams = 1
    elif n_grams == 'Two Word Phrases':
        n_grams = 2
    elif n_grams == 'Three Word Phrases':
        n_grams = 3

    submit4 = st.button('Analyze Text')

    if submit4:
        if len(st.session_state.text) > 0:
            st.write('I am sorry this method does not apply to single texts. Please return to the Home page and upload a CSV file of mutiple texts.')
        elif len(st.session_state.multitext_all) > 0:
            corpus = textacy.Corpus("en_core_web_sm", data=st.session_state.multitext_all)
            docs_terms = (textacy.extract.terms(doc, ngs=partial(textacy.extract.ngrams, n=n_grams, include_pos={"PROPN", "NOUN", "ADJ", "VERB"})) for doc in corpus)
            tokenized_docs = (textacy.extract.terms_to_strings(doc_terms, by="lemma") for doc_terms in docs_terms)
            doc_term_matrix, vocab = textacy.representations.build_doc_term_matrix(tokenized_docs, tf_type='linear', idf_type='smooth')
            model = textacy.tm.TopicModel(model_type, n_topics=num_topics)
            model.fit(doc_term_matrix)
            doc_topic_matrix = model.transform(doc_term_matrix)
            id_to_term = {id_: term for term, id_ in vocab.items()}
            st.write('--------------------------------------------')
            st.write('**Below are the top 10 word/phrases for each topic:**')
            for topic_idx, terms in model.top_topic_terms(id_to_term, top_n=words_in_topic):
                st.write(f"**Topic {topic_idx}**: {'; '.join(terms)}")
            st.write('--------------------------------------------')
            st.write('**Chart of Topics and Words**')
            plot1 = model.termite_plot(doc_term_matrix, id_to_term, n_terms=12, highlight_topics=graph_topic_to_highlight,save = "termite_plot.png")
            st.image('./termite_plot.png')
        else:
            st.write("Please upload a CSV file on the Home page.")
    
if selected == 'Document Clustering':
    
    st.subheader("Document Clustering")
    st.write("**Description:** This task involves placing documents into groups based on similarity and then extracting the key words/phrases from each group.")
    st.write("_Note: This task only applies to multiple document files (i.e. an uploaded CSV file)._")
    API_URL = "https://api-inference.huggingface.co/models/Craig/paraphrase-MiniLM-L6-v2"
    ensure_model_is_ready = query({"inputs": ['test 1','test 2'], "parameters": {"wait_for_model": True}})
    
    num_clusters = st.slider('Number of Clusters to Create ', min_value=2, max_value=10, value=4, step=1)
    submit6 = st.button('Analyze Text')

    if submit6:
        if len(st.session_state.text) > 0:
            st.write('I am sorry this method does not apply to single texts. Please return to the Home page and upload a CSV file of mutiple texts.')
        elif len(st.session_state.multitext_all) > 0:
            list_of_text = st.session_state.multitext_all
            corpus_embeddings = query({"inputs": list_of_text, "wait_for_model": True})
            
            clustering_model = KMeans(n_clusters=num_clusters, random_state=15)
            clustering_model.fit(corpus_embeddings)
            cluster_assignment = clustering_model.labels_
            cluster_assignment += 1

            df1 = pd.DataFrame(list(zip(list_of_text, cluster_assignment)), columns=['text','cluster'])
            model = 'all-MiniLM-L6-v2'
            kw_model = KeyBERT(model)

            kp_vect = KeyphraseCountVectorizer(lowercase=False)

            lister = []
            for i in df1.cluster.unique():
                lister.append(i)
                lister.sort()

            lister2 = []
            for i in lister:
                lister2.append(' '.join(df1[df1.cluster == i]['text'].tolist()))
            clust = 1
            new_list = []
            my_bar2 = st.progress(0)
            for i in lister2:
                keyphrase_data = kw_model.extract_keywords(docs=i, vectorizer=kp_vect, top_n=6, use_maxsum=True, nr_candidates=18)
                phrases =[]
                for i in keyphrase_data:
                    phrases.append(i[0])
                new_list.append(phrases)
                my_bar2.progress(round(clust/len(lister2), 1))
                clust +=1

            joined_list = []
            for i in new_list:
                joined_list.append(' ; '.join(i))
            df4 = pd.DataFrame(list(zip(lister, joined_list)), columns=['Cluster Number','Key Words/Phrases Within Each Cluster'], index=None)
            df4.index = [""] * len(df4)
            st.write('The table below displays the document clusters and key words/phrases extracted from each cluster.')
            st.table(df4)
            embeds = pd.DataFrame(corpus_embeddings)
            pca = PCA(n_components=3)
            pca_test = pca.fit_transform(embeds)

            cluster_assignment = [str(x) for x in cluster_assignment]
            new_list1 = []
            pca_outputs = pd.DataFrame(pca_test)
            pca_outputs.columns = ['First Component', 'Second Component', 'Third Component']
            pca_outputs['cluster'] = cluster_assignment
            pca_outputs['text'] = list_of_text
            test = [new_list1.append(i[:60]+"...") for i in list_of_text]
            pca_outputs['short_text'] = new_list1
            pca_outputs.sort_values(by=['cluster'], ascending=True, inplace=True)

            st.write('--------------------------------------------')

            st.write('Below is an interactive 2D scatter plot of the documents. Each point represents a document and the colors correspond to the clusters. Click and drag over several points to see the text displayed on the right.')

            pipe = make_pipeline(TfidfVectorizer(), UMAP(random_state=42))
            X = pipe.fit_transform(st.session_state.multitext_all)
            temp_chart = plot_text(X, st.session_state.multitext_all, color_array=pca_outputs.cluster)
            st.altair_chart(temp_chart, use_container_width=True)

            st.write('--------------------------------------------')

            fig = px.scatter_3d(pca_outputs, x='First Component', y='Second Component', z='Third Component', color='cluster', hover_name='short_text')
            st.write('Below is an interactive 3D scatter plot of the documents. Each point represents a document and the colors correspond to the clusters. You can explore the graphic closer by clicking the "view fullscreen" button in the top right corner.')
            st.plotly_chart(fig)
            cluster_df = pca_outputs[['cluster', 'text']]
            Cluster_Results = cluster_df.to_csv(index=False, header=True)
            st.download_button("Download Results", Cluster_Results, file_name="Document_Clustering_Results.csv")

        else:
            st.write("Please upload a CSV file on the Home page.")

#### This last portion is something of a placeholder -- the code allows putting //
# a small amount of text at the bottom of the page and could be usefil later on //
# if this application is deployed more widely.

hide_streamlit_style = """
            <style>
            footer {visibility: hidden;}
            footer:after {
	        content:'Created by Adam J'; 
	        visibility: visible;
	        display: block;
	        position: relative;
	        padding: 5px;
	        top: 2px;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
