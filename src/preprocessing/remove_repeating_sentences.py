import pandas as pd
import numpy as np


import nltk
from sklearn.feature_extraction.text import TfidfVectorizer


def remove_repeating_greetings(df: pd.DataFrame, text_column: str = "translatedText", percentile: float = 6.2) -> pd.DataFrame:
    df[text_column] = df[text_column].str.strip()
    df = df[df[text_column].str.len() != 0]  # remove empty speeches

    sentence_list = df[text_column].apply(lambda txt: nltk.sent_tokenize(txt))

    print("Fitting TF-IDF vectorizer...")
    vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    min_df=2,       
    max_df=0.99
    )

    tfidf = vectorizer.fit(sentence_list[sentence_list.str.len() != 0].explode())


    first_sentences = sentence_list.apply(lambda lst: lst[0] if len(lst) else "")
    print(f"Extracted {len(first_sentences)} first sentences")

    X = tfidf.transform(first_sentences)
    sentence_scores = np.asarray(X.mean(axis=1)).flatten()
    cut_off = np.percentile(sentence_scores, percentile)

    remove_mask = sentence_scores <= cut_off
    

    print(f"Removing {percentile:.2f}% of first sentences")

    df.loc[remove_mask, text_column] = sentence_list[remove_mask].apply(lambda sentences: " ".join(sentences[1:]))
    
    df = df[df[text_column].str.len() != 0]

    return df


def remove_repeating_endings(df: pd.DataFrame, text_column: str = "translatedText", percentile: float = 4.2) -> pd.DataFrame:

    df[text_column] = df[text_column].str.strip()
    df = df[df[text_column].str.len() != 0]  # remove empty speeches

    sentence_list = df[text_column].apply(lambda txt: nltk.sent_tokenize(txt))

    print("Fitting TF-IDF vectorizer...")
    vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    min_df=2,       
    max_df=0.99
    )


    tfidf = vectorizer.fit(sentence_list[sentence_list.str.len() != 0].explode())


    first_sentences = sentence_list.apply(lambda lst: lst[-1] if len(lst) else "")
    print(f"Extracted {len(first_sentences)} last sentences")

    X = tfidf.transform(first_sentences)
    sentence_scores = np.asarray(X.mean(axis=1)).flatten()
    cut_off = np.percentile(sentence_scores, percentile)

    remove_mask = sentence_scores <= cut_off
    

    print(f"Removing {percentile:.2f}% of last sentences")

    df.loc[remove_mask, text_column] = sentence_list[remove_mask].apply(lambda sentences: " ".join(sentences[:-1]))
    
    df = df[df[text_column].str.len() != 0]

    return df