# euparl_discourse

### Translation
- send_translation_requests.py: To avoid Gemini's rate limits, translation is sent in batches of varying sizes, retrying with a smaller batch size after failure. This is semi-automatic so that once no requests are possible anymore due to rate limits, one has to restart later at the point of last successful iteration. 
- process_translations.py: Once all requests are sent, load and process the model's responses and create a new dataframe with the translated speeches


### LDA
- first create multiple LDA models using scripts/lda/prepare_and_fit_multiple_lda.py => they will be saved in data/lda/{x}_topics/{k_passes}/model.model; this will also preprocess (or load already preprocessed data) data and dictionary 
- then evaluate (using scripts/lda/evaluate_lda.py), this will chose model with best coherence which has a topic related to migration (also inspect manually the created topics of the chosen model to make sure they make sense); the chosen model is saved in data/lda/final/model.model; remember which number of topics and passes was selected 
- speeches are assigned to topics in preprocessing 