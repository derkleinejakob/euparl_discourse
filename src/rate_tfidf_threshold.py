import pandas as pd
import numpy as np
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from ipywidgets import Button, Output
from IPython.display import display
import textwrap


def rate_tfidf_threshold(df, score_range=(5,15), n_samples=100, min_df=2, max_df=0.99):
    """
    Identify sentences with TF-IDF scores in a specified range and sample them for manual rating.
    
    Args:
        df: DataFrame containing text data
        score_range: Tuple of (min_percentile, max_percentile) to filter sentences by TF-IDF score
        n_samples: Number of sentences to sample for manual rating
        min_df: Minimum document frequency for TF-IDF (ignore terms appearing in fewer documents)
        max_df: Maximum document frequency for TF-IDF (ignore terms appearing in more than this proportion)
    
    Returns:
        DataFrame with sampled sentences, their scores, percentiles, and ratings
    """
    
    # Step 1: Extract first sentences from all documents
    print("Tokenizing sentences...")
    text_column = "translatedText"
    
    # Collect the first sentence from each document
    all_sentences_list = []
    for idx, txt in enumerate(df[text_column]):
        # Skip empty or invalid entries
        if pd.notna(txt) and isinstance(txt, str) and len(txt.strip()) > 0:
            # Split text into sentences using NLTK
            sentences = nltk.sent_tokenize(txt)
            # Extract only the first sentence if it exists and is non-empty
            if sentences and len(sentences[0].strip()) > 0:
                all_sentences_list.append(sentences[0])
    
    print(f"Found {len(all_sentences_list)} first sentences")
    
    if len(all_sentences_list) == 0:
        print("No valid sentences found!")
        return None
    
    # Step 2: Create TF-IDF model to score sentences
    print("Fitting TF-IDF vectorizer...")
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),  # Use both single words and two-word phrases
        min_df=min_df,       # Ignore rare terms
        max_df=max_df        # Ignore very common terms
    )
    # Train the vectorizer on all first sentences
    tfidf = vectorizer.fit(all_sentences_list)
    
    # Step 3: Calculate TF-IDF scores for all sentences
    print("Calculating TF-IDF scores...")
    X = tfidf.transform(all_sentences_list)  # Transform sentences to TF-IDF vectors
    sentence_scores = np.asarray(X.mean(axis=1)).flatten()  # Average TF-IDF across all terms in each sentence
    
    # Step 4: Filter sentences by specified percentile range
    # Convert percentile range to actual score thresholds
    min_score, max_score = np.percentile(sentence_scores, score_range[0]), np.percentile(sentence_scores, score_range[1])
    # Create boolean mask for sentences within the score range
    mask = (sentence_scores >= min_score) & (sentence_scores <= max_score)
    
    # Create DataFrame containing only sentences within the target score range
    candidates = pd.DataFrame({
        'sentence': [all_sentences_list[i] for i in range(len(all_sentences_list)) if mask[i]],
        'score': sentence_scores[mask]
    })
    
    if len(candidates) == 0:
        print("No sentences found in the specified score range!")
        return None
    
    # Step 5: Sample sentences for manual rating
    # Randomly sample up to n_samples sentences from candidates
    sampled = candidates.sample(n=min(n_samples, len(candidates))).reset_index(drop=True)
    sampled['rating'] = None  # Placeholder for manual ratings
    # Calculate what percentile each sentence's score represents
    sampled['percentile'] = sampled['score'].apply(lambda x: (sentence_scores < x).sum() / len(sentence_scores) * 100)
    
    # Step 6: Set up interactive rating interface
    ratings = []  # Store ratings as they're collected
    current_index = [0]  # Track which sentence we're currently rating (list to allow modification in nested function)
    
    print(f"\n{'='*90}")
    print(f"Starting manual rating: {len(sampled)} sentences")
    print(f"{'='*90}\n")
    
    # Create output widget to display sentences
    output = Output()
    display(output)
    
    def show_sentence(index):
        """Display a sentence and its metadata for rating"""
        with output:
            output.clear_output(wait=True)  # Clear previous sentence display
            
            # Check if we've rated all sentences
            if index >= len(sampled):
                print(f"Rating complete! Rated {len(ratings)}/{len(sampled)} sentences.")
                # Add collected ratings to the dataframe
                sampled.loc[:len(ratings)-1, 'rating'] = ratings
                print(f"\nFinal ratings: {ratings}")
                return
            
            # Display current sentence with metadata
            row = sampled.iloc[index]
            print(f"Progress: {index+1}/{len(sampled)} sentences rated")
            print(f"\nSentence {index+1}/{len(sampled)}")
            print(f"TF-IDF score: {row['score']:.6f}")
            print("-" * 90)
            print(textwrap.fill(row['sentence'], width=90))  # Wrap text for readability
            print("-" * 90)
    
    def on_button_click(button, rating_value):
        """Handle button clicks to record ratings and advance to next sentence"""
        ratings.append(rating_value)  # Store the rating
        sampled.loc[current_index[0], 'rating'] = rating_value  # Add rating to dataframe
        current_index[0] += 1  # Move to next sentence
        show_sentence(current_index[0])  # Display next sentence
    
    # Create rating buttons
    btn_correct = Button(description="1: Remove")
    btn_substantive = Button(description="0: Substantive")
    
    # Attach click handlers with corresponding rating values
    btn_correct.on_click(lambda b: on_button_click(b, 1))
    btn_substantive.on_click(lambda b: on_button_click(b, 0))
    
    # Display first sentence and buttons
    show_sentence(0)
    display(btn_correct, btn_substantive)
    
    # Return the sampled dataframe with ratings
    return sampled