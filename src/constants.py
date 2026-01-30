from tueplots.constants.color import rgb

PATH_RAW_DATA = "data/intermed/speech_output.csv"
PATH_TRANSLATED_DATA = "data/intermed/speech_translated.parquet"
PATH_DF_TRANSLATION_TEST = "data/translation/df_translation_test.parquet"

PATH_ALL_SPEECHES = "data/final/full.parquet" # formerly known as final.parquet
PATH_MIGRATION_SPEECHES = "data/final/migration.parquet" 
PATH_MIGRATION_CHES = "data/final/migration_with_ches.parquet"
PATH_MIGRATION_SPEECHES_EMBEDDED = "data/final/migration_with_embeddings.parquet" # formerly known as SPEECH_EMBEDDINGS.parquet
PATH_MIGRATION_SPEECHES_SIMILARITIES = "data/final/migration_with_similarities.parquet"
PATH_VOCAB_EMBEDDED = "data/final/vocab_embeddings.parquet" # formerly known as VOCAB_EMBEDDGINGS.parquet
PATH_MODEL = "data/lda/final_model/model.model"

COLOR_MAP_PARTY = {
    "PSE/S\&D": "#E41A1C",
    
    "Greens/EFA": "#32CD32",

    "ELDR/ALDE/Renew": "#FFD700",     

    "PPE": "#1F77B4",      
    "PPE-DE": "#1E90FF",   
    "UEN": "#F4D03F",      

    "ECR": "#0057A4",      
    "EDD/INDDEM/EFD": "#6A5ACD",  
    "EFD": "#7B68EE",      
    "EFDD": "#9370DB",     

    "ITS": "#00008B",     
    "ENF/ID": "#000080",      

    "NGL/The Left": "#8B0000",
}

# use Uni Tübingen corporate design? 
COLOR_MAP_BLOCK = { 
    "left": rgb.tue_violet,
    "green": rgb.tue_green,
    "social_democratic": rgb.tue_red,
    "christian_conservative": "black",
    "liberal": rgb.tue_orange,
    "(extreme)_right": rgb.tue_blue,
}

ORDER_BLOCK = [
    "(extreme)_right",
    "christian_conservative",
    "liberal",
    "social_democratic",
    "green",
    "left"
]

LEGEND_BLOCK = {
    "left": "Left",
    "green": "Greens",
    "social_democratic": "Social Democrats",
    "christian_conservative": "Conservatives",
    "liberal": "Liberals",
    "(extreme)_right": "(Far) Right",
}

COLOR_MAPS = {
    "party": COLOR_MAP_PARTY,
    "block": COLOR_MAP_BLOCK,
}

ELECTION_YEARS = [1999, 2004, 2009, 2014, 2019, 2024]

EMBEDDING_MODEL = "google/embeddinggemma-300m"

N_TOPICS = 30

MIGRATION_THRESHOLD = 0.25
MIGRATION_TOPIC_ID = 19
TOPIC_LABELS = [
    "EU Security / Defense", # 0
    "Debate Etiquette / Brexit", # 1
    "EU Finances", # 2
    "Workers / Industry", # 3
    "Fishing", # 4
    "Budgetary Control", # 5
    "Economic Development", # 6
    "Human Rights", # 7
    "Rule of Law", # 8
    "Taxation", # 9
    "Gender Equality", # 10
    "Terror / Political Violence", # 11
    "Food Safety", # 12
    "Economic Crisis", # 13
    "Climate / Energy", # 14
    "Trade Relations", # 15
    "International Conflicts", # 16
    "Education / Culture", # 17
    "Intra-European disputes", # 18
    "Migration / Asylum", # 19
    "Legislative Process", # 20
    "Russia / Ukraine", # 21
    "Social Policy / Labor", # 22
    "Data Protection", # 23
    "Agriculture", # 24
    "Election Law", # 25
    "Market Regulation", # 26
    "Disasters / Epidemics", # 27
    "Sanctions / Condemnations", # 28
    "Children’s Rights" # 29
]

TOPIC_WORDS = ['security, united, common',
 'want, think, commissioner',
 'budget, fund, financial',
 'worker, transport, sector',
 'fishing, fishery, sea',
 'financial, committee, agency',
 'development, economic, strategy',
 'human, freedom, democracy',
 'law, rule, government',
 'tax, company, fraud',
 'woman, gender, equality',
 'attack, world, today',
 'health, substance, china',
 'bank, crisis, economic',
 'energy, climate, emission',
 'agreement, trade, council',
 'conflict, peace, humanitarian',
 'education, cultural, human',
 'turkey, germany, minister',
 'refugee, border, migration',
 'resolution, group, text',
 'ukraine, russia, russian',
 'social, poverty, worker',
 'datum, cooperation, information',
 'product, food, agricultural',
 'political, election, process',
 'market, regulation, service',
 'health, disaster, cause',
 'sanction, death, penalty',
 'child, convention, international']

CHES_DIMENSIONS = [
    "lrgen",
    "lrecon",
    "lrecon_salience",
    "lrecon_dissent",
    "lrecon_blur",
    "galtan",
    "galtan_salience",
    "galtan_dissent",
    "galtan_blur",
    "eu_position",
    "eu_salience",
    "eu_dissent",
    "eu_blur",
    "spendvtax",
    "spendvtax_salience",
    "deregulation",
    "dereg_salience",
    "redistribution",
    "redist_salience",
    "econ_interven",
    "civlib_laworder",
    "civlib_salience",
    "sociallifestyle",
    "social_salience",
    "womens_rights",
    "lgbtq_rights",
    "samesex_marriage",
    "religious_principles",
    "relig_salience",
    "immigrate_policy",
    "immigrate_salience",
    "immigrate_dissent",
    "multiculturalism",
    "multicult_salience",
    "multicult_dissent",
    "nationalism",
    "nationalism_salience",
    "ethnic_minorities",
    "ethnic_salience",
    "urban_rural",
    "urban_salience",
    "environment",
    "enviro_salience",
    "climate_change",
    "climate_change_salience",
    "protectionism",
    "regions",
    "region_salience",
    "international_security",
    "international_salience",
    "us",
    "us_salience",
    "eu_benefit",
    "eu_ep",
    "eu_fiscal",
    "eu_intmark",
    "eu_employ",
    "eu_budgets",
    "eu_agri",
    "eu_cohesion",
    "eu_environ",
    "eu_asylum",
    "eu_foreign",
    "eu_turkey",
    "eu_russia",
    "russian_interference",
    "anti_islam_rhetoric",
    "people_vs_elite",
    "antielite_salience",
    "corrupt_salience",
    "members_vs_leadership",
    "executive_power",
    "judicial_independence",
]
