import os 
from gensim import corpora
from gensim.models import LdaMulticore
from gensim.utils import simple_preprocess
import spacy
import logging
from typing import List
from tqdm import tqdm
import json 
import pandas as pd
import os 
import sys
from pathlib import Path

# assume script is run from project root => path to be able to import src
sys.path.append(str(Path.cwd()))
from src.constants import PATH_ALL_SPEECHES


def preprocess_documents(documents: List[str], custom_stopwords=[], test_first_k = None):     
    logging.basicConfig(format ='%(asctime)s : %(levelname)s : %(message)s')
    logging.root.setLevel(level = logging.WARN)
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

    def preprocess_document(document):    
        # tokenize using gensim's default preprocessing
        tokens = simple_preprocess(document)
        document = nlp(" ".join(tokens))
        # lemmatize and remove stopwords 
        lemmas = [token.lemma_ for token in document if (not token.is_stop) and (not token.lemma_ in custom_stopwords)]
        return lemmas

    if test_first_k: 
        documents = documents[:test_first_k]
    
    processed_data = [preprocess_document(doc) for doc in tqdm(documents, "preprocessing")]
    return processed_data
    

def get_preprocessed_documents(preprocessed_full_path, df = None): 
    if os.path.exists(preprocessed_full_path): 
        print("Loading preprocessed data", preprocessed_full_path)
        preprocessed_data = json.load(open(preprocessed_full_path))
        # preprocessed_data = [preprocessed_data[i] for i in df_party_members.index.tolist()]  # align with filtered dataframe
        json.dump(preprocessed_data, open(preprocessed_full_path, "w+"))
    else: 
        # if not exists: preprocess once and keep json 
        preprocessed_data = preprocess_documents(documents = df["translatedText"].tolist())
        json.dump(preprocessed_data, open(preprocessed_full_path, "w+"))
    
    return preprocessed_data 

def fit_models(corpus, dictionary, n_topic_values = {50: [5, 7, 10], 60: [5], 80: [5], 100: [5], 120: [5]}, n_workers = 6):
    runs = []
    for n_topics, n_passes_values in n_topic_values.items(): 
        for n_passes in n_passes_values: 
            os.makedirs(f"data/lda/screens/{n_topics}_topics/{n_passes}", exist_ok=True)
            out_path = f"data/lda/screens/{n_topics}_topics/{n_passes}/model.model"
            workers = n_workers

            print("Fitting model with", n_topics, "topics and", n_passes, "passes")
            lda_model = LdaMulticore(corpus = corpus, id2word=dictionary, num_topics = n_topics, passes = n_passes, workers=workers)
            lda_model.save(out_path)
            
            runs.append((n_topics, n_passes, lda_model))
    return runs 

if __name__ == "__main__": 
    """
    Creates multiple LDA models for different topic values and n_passes
    """

    PATH_PREPROCESSED = "data/lda/preprocessed_texts_all_translated.json"
    PATH_DICTIONARY = "data/lda/dictionary_final.d"
    PATH_CORPUS = "data/lda/corpus_final.c"
    PATH_CONFIGS = "data/lda/screen_configs.json"

    df = pd.read_parquet(PATH_ALL_SPEECHES)

    preprocessed_data = get_preprocessed_documents(PATH_PREPROCESSED, df)

    configs = json.load(open(PATH_CONFIGS))

    print("Creating dictionary")
    dictionary = corpora.Dictionary(preprocessed_data)
    print("Filtering dictionary")

    n_before_filtering = len(dictionary)

    # NOTE: these thresholds were identified manually => see the corresponding notebook to see how
    
    dictionary.filter_extremes(
        no_below=10,     # Keep tokens appearing in at least 10 speeches
        no_above=0.152,    # Remove tokens appearing in more than 15.2% of speeches
        keep_n=100000    # Here: keep all tokens, because there are only 59173 words in the dictionary
    )

    n_after_filtering = len(dictionary)
    print("Filtered dictionary: ")
    print("Before:",n_before_filtering)
    print("Now:", n_after_filtering, f"{'%.2f' % (n_after_filtering / n_before_filtering)}") 

    corpus = [dictionary.doc2bow(l) for l in tqdm(preprocessed_data, "Preparing corpus")]

    dictionary.save(PATH_DICTIONARY)
    print("Saved dictionary to", PATH_DICTIONARY)
    corpora.MmCorpus.serialize(PATH_CORPUS, corpus)
    print("Saved corpus to", PATH_CORPUS)

    
    fit_models(corpus, dictionary, configs)