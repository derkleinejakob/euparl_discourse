# %%
import json 
import pandas as pd 
from google import genai 
import os 
from tqdm import tqdm 
import optparse

import sys
from pathlib import Path

# assume script is run from project root => path to be able to import src
sys.path.append(str(Path.cwd()))
from src.constants import PATH_RAW_DATA, PATH_TRANSLATED_DATA, PATH_DF_TRANSLATION_TEST

def load_job(job_file_path, results_path): 
    """Assumes the job_name is saved in JOB_FILE_PATH. Downloads Gemini's responses if job is done and saves them in a RESULTS_PATH jsonl file"""
    
    global client
    if client is None: 
        api_key = os.environ.get('GOOGLE_API_KEY')
        assert api_key is not None, "Missing API key"
        client = client if client is not None else genai.Client(api_key=api_key)

    job_name = json.load(open(job_file_path))["job_name"]
    batch_job = client.batches.get(name=job_name)
    if batch_job.state.name == "JOB_STATE_SUCCEEDED":        
        result_file_name = batch_job.dest.file_name
        file_content_bytes = client.files.download(file=result_file_name)
        content = file_content_bytes.decode('utf-8') 
        
        if content: 
            with open(results_path, "w+") as f: 
                f.write(content)
        return content 
    else: 
        print(batch_job.state.name)
    
def process_responses(results_path):
    """
    Assumes the job is already downloaded using load_job() and results are in RESULTS_PATH: 
    creates a dictionary that maps request_ids (consisting of index_date_speechnumber) to the LLM response
    """    
    with open(results_path) as f: 
        lines = f.readlines()   

    translations = dict()
    total_tokens = 0 
    for line in lines: 
        if line: 
            line_obj = json.loads(line)
            id = line_obj["id"]
            if "response" not in line_obj: 
                print("No response generated. Skipping", id)
                print(line_obj)
                continue
            response = line_obj["response"]
            total_tokens += response["usageMetadata"]["totalTokenCount"]
            assert len(response["candidates"]) == 1
            assert response["candidates"][0]["finishReason"] == "STOP"
            
            if "parts" not in response["candidates"][0]["content"]: 
                print("Warning: model responded with empty text", id)
                print(line_obj)
                translations[id] = ""
                continue
            
            assert len(response["candidates"][0]["content"]["parts"]) == 1
            response_text = response["candidates"][0]["content"]["parts"][0]["text"]
            
            translations[id] = response_text
    return (translations, total_tokens)

def add_translations_to_df(df, translations, is_test=False): 
    """After mapping the responsed to their ids, add the results to the dataframe"""
    for id in translations: 
        assert len(id.split("_")) == 4
        r, index, date, speechnum = id.split("_")
        
        col_text = "translatedText" if not is_test else "translationTest"
        col_source = "translationSource" if not is_test else "translationTestSource"

        index = int(index)
        speechnum = int(speechnum)
        # sanity checks to be absolutely sure the mapping of original speech => translation is correct
        assert df.iloc[index]["date"] == date
        assert df.iloc[index]["speechnumber"] == speechnum
        
        if "no translation needed" == translations[id]: 
            # text was in English already, copy it to translatedText
            df.at[index, col_text] = df.iloc[index]["text"]
            # keep track that Gemini assumed the text was in English 
            df.at[index, col_source] = "original_gm"
        else: 
            if "no translation needed" in translations[id]: 
                # sanity check that if no translation is needed model really only has returned "no translation needed" without anything else.
                print("Warning: gemini response contained no translation sentence but also something else")
                print(translations[id])
                
            df.at[index, col_text] = translations[id]
            df.at[index, col_source] = "machine_gm"


if __name__ == "__main__": 
    optParser = optparse.OptionParser()
    optParser.add_option('-t', '--test',action='store_true',
                         default=False, 
                         dest='test',
                         help='Whether to include translation tests')

    opts, _ = optParser.parse_args()
    include_tests = opts.test
    path_out = PATH_TRANSLATED_DATA
    
    #%%
    print("Reading data")
    df = pd.read_csv(PATH_RAW_DATA)
    df = df.reset_index()
    # %%
    # create new column translationSource
    df["translationSource"] = None 
    # if in parllaw, there was no translation needed, the translation in translatedText is the original speech 
    df.loc[df['translationInSpeech'] == False, 'translationSource'] = "original_pl"
    # parllaw created machine translation: 
    df.loc[df['translationInSpeech'] == True, 'translationSource'] = "machine_pl"

    if include_tests: 
        path_out = PATH_DF_TRANSLATION_TEST
        df["translationTest"] = None
        df["translationTestSource"] = None

    #%% 
    pending = 0 
    total_tokens = 0
    for root, dirs, files in os.walk("data/translation/job/"): 
        # iterate over all job files
        for f in tqdm(files): 
            assert f[:4] == "job-"
            assert f[-5:] == ".json"
            job_name = f[4:]
            job_name = job_name[:-5]
            
            is_test = "test" in job_name 
            if (is_test and not include_tests) or (not is_test and include_tests): 
                print("Skipping job:", job_name)
                continue

            job_file_path = root+"/"+f
            results_path = "data/translation/results/results-"+job_name+".jsonl"
            
            if not os.path.exists(results_path): 
                # potentially results were already downloaded previously, skip if already exists
                j = load_job(job_file_path, results_path)
                if j is None: 
                    # job is not done yet, have to wait for results
                    pending += 1
                    continue
            translations, tokens = process_responses(results_path)
            total_tokens += tokens
            add_translations_to_df(df, translations, is_test)
            
    print("pending:", pending)
    print("total tokens:", (total_tokens/10e5),"million")

    if pending > 0: 
        print("Wait for pending jobs")
        exit(0)
    if opts.test:
        df = df[~(df["translationTest"].isna())]
    else: 
        missing_transl = df[~(df["party"] == "-") & (df["translatedText"].isna())]
        translated = df[df["translationSource"].isin(["machine_gm", "original_gm"])]

        not_translated = missing_transl[missing_transl["translationSource"].isna()]
        print("still missing translation: ", len(not_translated))

    # %%
    df.drop("translationInSpeech", axis=1, inplace=True)
    df.drop("index", axis=1, inplace=True)
    df.to_parquet(path_out, index=False)
    print("Written data to", path_out)