# Outline for merging the ParlLawSpeech (PLS) Migration dataset with the CHES dataset

Since the PLS migration dataset does not come with metadata regarding the MEP's national party affiliation at a given
date on which a speech was held, we need to fetch additional data from the EU OpenData Portal (EU-ODP). The subsequent merge
with the CHES dataset then requires a mapping between national party IDs of the EU-ODP and the party IDs inherent to
the CHES dataset. For this mapping, just looking at the (raw) CHES dataset itself is not sufficient, since it only 
contains (CHES) party IDs and (one of possibly multiple) party label per datapoint (the survey/expert estimates on a 
certain year for a specific party). Therefore, the overall merging process will be conducted in the following steps:

0. Create a "CHES meta" dataset which parses the CHES codebook.pdf into a dataset containing **party ID**, all possible **party labels**
and all possible **party names** as well as the **country code** (to which the party belongs, in ISO format) of each national party
included in the CHES dataset
1. Enrich the PLS Migration Dataset with metadata of the speakers (fetched from EU-ODP), in particular their **person ID** (*"ep_identifier"*)
2. Create a "membership" dataset that maps for each speaker of the migration dataset (given their ep_identifier), their national party
affiliations at given time periods to metadata of the corresponding national party, in particular their **party labels** and **party names**
   (including available variants of these, e.g. english and native spellings/versions)
3. Create a mapping/enrich the *membership dataset* with the metadata of the *CHES_meta dataset*
4. Create a mapping/enrich the *PLS migration dataset* with the enriched *membership dataset* from the previous step
5. Create a final *"migration_ches_survey"* dataset by replacing/mapping the CHES information for each membership with/to the actual
survey items of the (raw) CHES dataset. Do this either by only merging speech and CHES score which have the exact same year (*default*) and/or
also merge all remaining speeches to the last (i.e. most recent past) CHES score available for the party, if applicable (*fallback* -> experimental) 
6. ... or instead, for the remaining CHES scores that have no exact match, linearly interpolate the CHES scores from two bracketing years/scores if applicable, i.e. for speech_2016 use the scores from 2014 and 2019 (if existing) to interpolate
   (*interpolated* -> experimental)

The outlined steps are handled by the following scripts, and each yields as output the input for the next step of the pipeline:
0. *experiments\preprocessing_checks\pre5_0_parse_ches_meta.ipynb*
1. *experiments\preprocessing_checks\pre5_1_enrich_migration_with_epID.ipynb*
2. *experiments\preprocessing_checks\pre5_2_create_nat_memberships.ipynb*
3. *experiments\preprocessing_checks\pre5_3_enrich_memberships_w_CHES.ipynb*
4. *experiments\preprocessing_checks\pre5_4_map_migration_to_nat_parties.ipynb*
5. *experiments\preprocessing_checks\pre5_5_replace_meta_w_raw_CHES.ipynb*
6. *experiments\preprocessing_checks\pre5_6_replace_w_interpolated_CHES.ipynb*








