#%% 
import pandas as pd
import json
from google import genai
import os 
from tqdm import tqdm
from datetime import datetime
import optparse


import sys
from pathlib import Path

# assume script is run from project root => path to be able to import src
sys.path.append(str(Path.cwd()))

from src.constants import PATH_RAW_DATA

# %% 
GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')
if GOOGLE_API_KEY is None: 
    raise ValueError("Missing API key")

MODEL_ID = "gemini-2.5-flash" 
PROMPT = "You are an expert translator for European Parliament speeches. Translate the following speech into English. Return only the translated text. Do not add explanations or comments. Preserve the original meaning and tone. If the speech is already in English, return 'no translation needed'."
client = genai.Client(api_key=GOOGLE_API_KEY)

#%%

def process_rows(df_missing, FILENAME, FROM_ROWS, TO_ROWS, c=True):
    """
    Create and upload requests for the passed dataframes, and create a batch job to start Gemini working on the requests.
    To control the batch size, use FROM_ROWS and TO_ROWS (will use df_missing[FROM_ROWS:TO_ROWS])
    
    :param df_missing: a dataframe with all speeches that are missing a translation
    :param FROM_ROWS: the index of row from the dataframe to start (inclusive)
    :param TO_ROWS: the index of row from the dataframe to end (exclusive)
    :param c: whether user has to confirm uploading the requests and creating a batch job for them by pressing enter
    """
    REQUESTS_PATH = f"data/translation/df/{FILENAME}.json"
    FILE_PATH = f"data/translation/file/file-{FILENAME}.json"
    JOB_FILE_PATH = f"data/translation/job/job-{FILENAME}.json"
    # keep track of failed jobs
    FAILED_TXT = f"data/translation/failed/names.txt"    
    
    def create_requests(df):
        """Create a jsonl file where each line is a request to the Gemini API""" 
        requests = [] 
        for _, row in df.iterrows():
            # as a key for the request, keep the index (row.name) but also for sanity check date and speechnumber
            key = f"r_{row.name}_{row['date']}_{row['speechnumber']}" 
            speech = row["text"]
            request = {
                "id": key, 
                "request": {
                    "contents": [{
                        "parts": [
                            {"text": f"{PROMPT}\n\n{speech}"},
                        ]
                    }]
                }
            }
            requests.append(request) 
        
        with open(REQUESTS_PATH, 'w') as f:
            for req in requests:
                f.write(json.dumps(req) + '\n')
        return requests

    def upload_batch(confirm=True): 
        """Upload the requests to Google (will not yet create a job for them). Assumes the requests are already saved in REQUESTS_PATH"""
        with open(REQUESTS_PATH) as f:
            n_requests = len(f.readlines())
            
        print("Sending",n_requests,"requests.")
        confirmed = input("Confirm with y:") if confirm else ""
        if confirmed == "": 
            print(f"Uploading file: {REQUESTS_PATH}")
            uploaded_batch_requests = client.files.upload(
                file=REQUESTS_PATH,
                config=genai.types.UploadFileConfig(display_name=f'{FILENAME}_{datetime.now()}')
            )
            json.dump({"file_name": uploaded_batch_requests.name}, open(FILE_PATH, "w+"))
            
            return uploaded_batch_requests
        else: 
            print("Terminated")
            # ideally, this would also remove the created files but since they are temporary anyway, skip this for now
            return None 
            
    def create_batch_job():
        """After uploading a file (with file_name on cloud saved in FILE_PATH), tell Gemini to process the file by creating a batch job."""
        def delete_file(path): 
            try:
                os.remove(path)
            except OSError:
                pass
            
        filename = json.load(open(FILE_PATH))["file_name"]
        
        try: 
            batch_job_from_file = client.batches.create(
                model=MODEL_ID,
                src=filename,
                config={
                    'display_name': f'job_{FILENAME}',
                }
            )
        except: 
            print("Could not create batch job.")
            print("Removing created files")
            delete_file(REQUESTS_PATH)
            delete_file(FILE_PATH)
            
            with open(FAILED_TXT, "a") as f:
                # append filename 
                f.write(FILENAME)
            return None 
        
        print(f"Created batch job from file: {filename}")
        # keep track of job_name in a json file 
        json.dump({
            "job_name": batch_job_from_file.name, 
        }, open(JOB_FILE_PATH, "w+"))
        return batch_job_from_file
        

    df_missing_translation = df_missing[FROM_ROWS:TO_ROWS]
    if len(df_missing_translation) == 0: 
        raise ValueError("Dataframe is empty, nothing left to translate")
    create_requests(df_missing_translation)
    uploaded_file = upload_batch(c)
    if not uploaded_file: 
        raise ValueError("Terminated upload")
    
    return create_batch_job()

if __name__ == "__main__": 
    optParser = optparse.OptionParser()
    optParser.add_option('-s', '--start_index',action='store', type='int',
                         dest='start_index',
                         help='Index to start sending next request from')
    optParser.add_option('-t', '--test',action='store_true',
                         default=False, 
                         dest='test',
                         help='Whether to send a test of 2000 samples which already have translations')
    
    optParser.add_option('-c', '--confirm',action='store_false',
                         default=True, 
                         dest='confirm',
                         help='Disable confirming each request by pressing enter.')

    opts, args = optParser.parse_args()
    os.makedirs("data/translation/df", exist_ok=True)
    os.makedirs("data/translation/failed", exist_ok=True)
    os.makedirs("data/translation/file", exist_ok=True)
    os.makedirs("data/translation/job", exist_ok=True)
    os.makedirs("data/translation/results", exist_ok=True)
    #%%
    print("Reading data")
    df = pd.read_csv(PATH_RAW_DATA)
    # keep track of original indices: 
    df = df.reset_index()


    # only translate speeches where speaker is member of a party because only these we need for our project
    df_missing = df[~(df["party"] == "-")]
    if opts.test:     
        # take a sample of 2000 speeches which already have a translation
        df_missing = df_missing[~(df_missing["translatedText"].isna())].sample(2000)
        print("n speeches which were in english", (df_missing["text"] == df_missing["translatedText"]).sum())
        print("n speeches which were not english", (~(df_missing["text"] == df_missing["translatedText"])).sum())
        start_index = 0 
        jobname = "df_test"
    else: 
        df_missing = df_missing[df_missing["translatedText"].isna()]
        jobname = "df"
        # NOTE: this needs to be done manually: 
        # start with index 0 and then start sending requests to the API 
        # once the rate limit is reached, wait for a while and continue with the requests. 
        # TODO: Make sure to update start_index to be the index of the first entry for which no job has been created yet.
        assert opts.start_index is not None 
        start_index = opts.start_index # start with 0


    running = True 
    # try sending batches of this size, but create smaller requests once they are rejected
    DEFAULT_BATCH_SIZE = 2000
    # whether to press enter before sending requests
    force_confirm = opts.confirm
    N_RETRIES = 5

    current_batch_size = DEFAULT_BATCH_SIZE
    retries = N_RETRIES

    while running: 
        if start_index > len(df): 
            print("Done.")
            break 

        if retries == 0: 
            print("Too many retries. Stopping")
            print("Continue next time at index", start_index)
            break 
        
        end_index = start_index + current_batch_size
        print("From", start_index,"to", end_index)

        filename = f"{jobname}_{start_index}_{end_index}"
        job = process_rows(df_missing, FILENAME=filename, FROM_ROWS=start_index, TO_ROWS=end_index, c=force_confirm)

        if job is None: 
            print("Job failed. Trying again with fewer requests")
            current_batch_size = current_batch_size // 2 
            retries -= 1
        else: 
            print("Job succeeded with size", current_batch_size)
            start_index = end_index
            # increase batch size gradually, but never exceed DEFAULT_BATCH_SIZE
            current_batch_size = min(current_batch_size*2, DEFAULT_BATCH_SIZE)
            # reset number of retries
            retries = N_RETRIES
    # %%
