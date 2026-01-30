#%% 
import pandas as pd 
from tqdm import tqdm
import optparse
import sys
from pathlib import Path

# assume script is run from project root => path to be able to import src
sys.path.append(str(Path.cwd()))

from preprocessing import add_party_orientation_year_agenda, keep_relevant_legislation_years, remove_commentary, remove_duplicate_speeches, remove_non_party_speeches, remove_repeating_greetings, remove_repeating_endings, rename_party_duplicates, assign_topics
from src.constants import PATH_TRANSLATED_DATA, PATH_ALL_SPEECHES, PATH_MIGRATION_SPEECHES, N_TOPICS
# TODO: run through all scripts in preprocessing folder and manipulate df, then output cleaned df
# TODO: is there a smarter way to do this that is less tedious? 
# TODO: remove empty text / text of certain length ? 

def save_df(dataframe, path): 
    if path.endswith(".csv"): 
        dataframe.to_csv(path)
    elif path.endswith(".parquet"): 
        dataframe.to_parquet(path)
    else: 
        raise ValueError(f"Unknown output file ending {path}")
    
def step(dataframe, process): 
    n_before = len(dataframe) 
    dataframe = process(dataframe)
    n_now = len(dataframe) 
    delta_n = n_before - n_now
    if not (delta_n == 0): 
        print(f"Removed {delta_n} rows ({'%.2f' % (delta_n / n_before)})")
    return dataframe

def main(): 
    optParser = optparse.OptionParser()
    optParser.add_option('-l', '--lda_finished', action='store_true',
                         default=False, dest='lda_finished',
                         help='After LDA is done, assign topic probabilities using the final LDA model')
    
    optParser.add_option('-t', '--topic_id', action="store",
                         default=19, dest="topic_id", help="Index of the migration topic in the final model")
    
    optParser.add_option('-r', '--relevance_threshold', action="store",
                         default=0.25, dest="relevance_threshold", help="Probability threshold to label a speech as covering migration")

    opts, _ = optParser.parse_args()

    print("Reading dataset")
    df = pd.read_parquet(PATH_TRANSLATED_DATA)
    # order of application matters! e.g. "rename party duplicates" should be run before "add party orientation blocks"
    preprocessing_steps = [remove_non_party_speeches, keep_relevant_legislation_years, remove_duplicate_speeches, add_party_orientation_year_agenda, rename_party_duplicates, remove_commentary, remove_repeating_greetings, remove_repeating_endings, assign_topics]

    n_columns_before = len(df.columns)
    n_before = len(df)
    print(f"Starting with {n_before} rows and {n_columns_before} columns")
    
    for process in tqdm(preprocessing_steps, "Preprocessing data"): 
        df = step(df, process)

    print(f"Done. Now have {len(df)} rows and {len(df.columns)} columns")
    
    if opts.lda_finished: 
        df["migration_prob"] = df[f"topic_{opts.topic_id}"]
        df_migration = df[df["migration_prob"] >= opts.relevance_threshold]
        df_migration = df_migration.drop(columns=[f"topic_{i}" for i in range(N_TOPICS)])
        save_df(df_migration, PATH_MIGRATION_SPEECHES)
    else: 
        print("Cannot create dataframe with only migration speeches yet. Run LDA first.")

    save_df(df, PATH_ALL_SPEECHES)

#%%
if __name__ == "__main__": 
    main()
# %%
