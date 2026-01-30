import textwrap
def print_top_speeches(df, topic_id, n_speeches=5):
    '''
    Print the n most probable speeches for a given topic id
    '''
    # get topic distribution for each document
    prob_column = "topic_" + str(topic_id)
    top_speeches = df.sort_values(by=prob_column, ascending=False).head(n_speeches)['translatedText']
    
    print(f"\nMost representative speeches for Topic {topic_id}:\n")
    for i, speech in enumerate(top_speeches):
        print(textwrap.fill(speech, width=80))
        print("\n" + "-"*80 + "\n")