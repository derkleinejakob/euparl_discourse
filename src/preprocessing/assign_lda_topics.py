from tqdm import tqdm 
from gensim import corpora
import json 
from gensim.models import LdaMulticore
import numpy as np 
import pandas as pd 
import os 
from src.constants import N_TOPICS

FINAL_MODEL_PATH = "data/lda/final_model/model.model"
PATH_CORPUS = "data/lda/corpus_final.c"

def assign_topics_(df, lda_model, n_topics, corpus): 
    def assign_topics(lda_model, corpus):
        topics = []
        for bow in tqdm(corpus, desc="Assigning most probable topic to each speech"):
            docs_topics = lda_model.get_document_topics(bow, minimum_probability=0)
            topics.append(docs_topics)
        return topics

    assert len(df) == len(corpus), "Number of rows and elements in the corpus do not match. Was the dataframe modified after LDA?"
    corpus_topics = assign_topics(lda_model, corpus)
    # set up df with topic probabilities

    # corpus topics is a list of lists of (topic_id, probability) tuples for each document in the corpus
    # this list is turned into dataframe of size (num_docs, num_topics) with probabilities
    num_docs = len(corpus_topics)
    
    topic_prob_matrix = np.zeros((num_docs, n_topics))
    for doc_idx, doc_topics in enumerate(corpus_topics):
        for topic_id, prob in doc_topics:
            topic_prob_matrix[doc_idx, topic_id] = prob
            
    topic_prob_df = pd.DataFrame(topic_prob_matrix, columns=[f"topic_{i}" for i in range(n_topics)], index=df.index)

    # append topic probabilities to df_party_members
    return pd.concat([df, topic_prob_df], axis=1)

def assign_topics(df): 
    # TODO: rerun LDA after data is cleaned and make sure preprocessed texts are now just aligned with the df

    if not os.path.exists(FINAL_MODEL_PATH): 
        print("No LDA model found. Not assigning topics yet. Create LDA model with intermediate dataset and find topic threshold, then re-run preprocessing.")
        return df 

    lda_model = LdaMulticore.load(FINAL_MODEL_PATH)

    # dictionary = corpora.Dictionary.load(PATH_DICTIONARY)
    corpus = corpora.MmCorpus(PATH_CORPUS)

    return assign_topics_(df, lda_model, N_TOPICS, corpus)