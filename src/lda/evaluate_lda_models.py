from gensim.models.coherencemodel import CoherenceModel
from gensim.models import LdaMulticore
from gensim import corpora
import json 
import os 

def evaluate_model(lda_model, n_topics, k_words, processed_texts, corpus, dictionary, search_term = "migration", compute_coherence=True): 
    """For the LDA model compute: 
    - coherence (a metric in LDA to express whether each word is associated with one topic (desireable, coherence => 1) or many (undesireable, coherence => 0)
    - highest probability that the search term is given in a topic
    - the most frequent position of the search term within the topics (e.g. if search term is most likely word in topic X, its most frequent position will be 0)
    - the indices of topics where the search term is within the k most likely words of that topic
    
    """
    if compute_coherence: 
        print("Computing coherence")
        coherence_model = CoherenceModel(
            model=lda_model, 
            texts=processed_texts,
            corpus=corpus, 
            dictionary=dictionary, 
            coherence='c_v'  # most common coherence measure
        )
        coherence_score = coherence_model.get_coherence()
    else: 
        coherence_score = None 
        
    
    # for each topic get probability of migration 
    # print k most likely words for 3 topics with highest probability 
    # return max probability and whether migration appeared in k most likely words of any topic 

    # find maximum probability of search term in the topics
    search_term_max_prob = float("-inf")
    search_term_highest_pos = float("inf")
    indices_relevant_topics = []
    
    for topic_index, topic in lda_model.show_topics(formatted=False, num_topics=n_topics):
        topic_words, topic_probs = zip(*topic)
    
        if search_term in topic_words: 
            idx = topic_words.index(search_term)
            search_term_max_prob = max(topic_probs[idx], search_term_max_prob)
            search_term_highest_pos = min(idx, search_term_highest_pos)        
            # check if search term appears in k most likely words (are ordered by their likelihood)
            if idx < k_words: 
                indices_relevant_topics.append(topic_index)  

                label = ", ".join([f"{word} ({'%.2f' % prob})" for word, prob in topic[:k_words]])
                print(f"Possibly relevant topic {topic_index}: {label}")
                
    print("."*30)
    print("Coherence:", coherence_score)
    print(f"Highest probability of {search_term}: {search_term_max_prob}")
    print(f"Most likely position of {search_term}: {search_term_highest_pos}")
    print(f"Relevant topics: {indices_relevant_topics} (n: {len(indices_relevant_topics)})")
    return coherence_score, search_term_max_prob, search_term_highest_pos, indices_relevant_topics

def print_topics(model, n_topics, k_words=10):
    for idx, topic in model.show_topics(formatted=False, num_topics=n_topics):
        label = ", ".join([f"{word} ({'%.2f' % prob})" for word, prob in topic[:k_words]])
        print(f"Topic {idx}: {label}")

if __name__ == "__main__": 
    MODELS_PATH = "data/lda/screens"
    PATH_DICTIONARY = "data/lda/dictionary_final.d"
    PATH_CORPUS = "data/lda/corpus_final.c"
    COMPARISON_RESULTS_PATH = "data/lda/screens/comparison.json"
    PATH_PREPROCESSED = "data/lda/preprocessed_texts_all_translated.json"

    K_WORDS = 10 # check for relevant keyword (e.g. migration) as being in most probable k words of the topic 

    dictionary = corpora.Dictionary.load(PATH_DICTIONARY)
    corpus = corpora.MmCorpus(PATH_CORPUS)
    processed_texts = json.load(open(PATH_PREPROCESSED))

    # os.makedirs(FINAL_MODEL_PATH, exist_ok=True)

    
    best_coherence = 0
    best_model = None 
    best_config = None 

    stats = []

    for root, folders_n_topics, _ in os.walk(MODELS_PATH): 
        for folder_n_topics in sorted(folders_n_topics): 
            try: 
                n_topics = int(folder_n_topics.split("_")[0]) # assume folder is named e.g. 50_topics
            except: 
                print("Could not read n_topics, skipping:", folder_n_topics)
                continue
            for _, folders_n_pases, _ in os.walk(os.path.join(root, folder_n_topics)): 
                for folder_n_passes in sorted(folders_n_pases): 
                    n_passes = int(folder_n_passes) # assume fjolder is just named e.g. 5 for 5 passes 

                    if not os.path.exists(os.path.join(root, folder_n_topics, folder_n_passes, "model.model")): 
                        print("Missing model", n_topics, "/", folder_n_passes)
                        continue
                    lda_model = LdaMulticore.load(os.path.join(root, folder_n_topics, folder_n_passes, "model.model"))
                    
                    print("\n"*2)
                    print("Evaluating model with", n_topics, "topics and", n_passes, "n_passes")

                    coherence, highest_prob, top_pos, indices_relevant_topics = evaluate_model(lda_model, n_topics, K_WORDS, processed_texts, corpus, dictionary)    
                    stats.append({
                        "n_topics": n_topics, 
                        "n_passes": n_passes,
                        "coherence": str(coherence),  
                        "highest_prob": str(highest_prob), 
                        "top_pos": top_pos, 
                        "n_relevant_topics": len(indices_relevant_topics)
                    })
                    if not (len(indices_relevant_topics) == 1): 
                        print("Ignoring model because it does not have exactly 1 relevant topic; has:", len(indices_relevant_topics))
                    else: 
                        if coherence is not None and coherence > best_coherence: 
                            best_coherence = coherence
                            best_model = lda_model 
                            best_config = (n_topics, n_passes)
        
    if best_config is not None: 
        print("Best model is", best_config, "with coherence:", best_coherence)
        print("Best model topics:")
        print_topics(best_model, best_config[0], K_WORDS)
    json.dump(stats, open(COMPARISON_RESULTS_PATH, "w"))