from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder

from src.constants import EMBEDDING_MODEL

from IPython.display import display, HTML
import pandas as pd
import numpy as np

def display_axis_semantics(axis_words: list[tuple[list[str]]]) -> None:
    data = []
    for axis, pairs in enumerate(axis_words):
        data.append({"Axis": axis, "Direction": "Neg", "Words": ", ".join([elem[0] for elem in pairs[0]])})
        data.append({"Axis": axis, "Direction": "Pos", "Words": ", ".join([elem[0] for elem in pairs[1]])})

    df = pd.DataFrame(data).set_index(["Axis", "Direction"])
    df['Words'] = df['Words'].str.wrap(100)
    display(HTML(df.to_html().replace("\\n","<br>")))


def closest_words_for_pc(k, model, vocab, probe_embs, top_n=20):
    reduced = model.transform(probe_embs)

    reduced = reduced 
    #/ np.linalg.vector_norm(reduced, ord=2, axis=1, keepdims=True)
    
    sorted_indices = np.argsort(reduced[:, k])
    pos_idx = sorted_indices[-top_n:]

    neg_idx = sorted_indices[:top_n]
    

    def map_indices_to_examples(index_list): 
        if isinstance(vocab, pd.DataFrame) or isinstance(vocab, pd.Series): 
            return [(vocab.iloc[i], reduced[:, k][i]) for i in index_list]
        else:
            return [(vocab[i], reduced[:, k][i]) for i in index_list]

    return map_indices_to_examples(neg_idx),  map_indices_to_examples(pos_idx)


def get_extreme_examples(df: pd.DataFrame, embeddings: np.array, top_k=10):
    
    return df.iloc[[embeddings[:, 0].argmin().item(),
                     embeddings[:, 0].argmax().item(),
                    embeddings[:, 1].argmin().item(),
                     embeddings[:, 1].argmax().item()]]

N_PCS = 100

def principle_component_regression(df: pd.DataFrame, target_var: str = "party",
                                   embedding_model: str = EMBEDDING_MODEL):

    X = np.stack(df[embedding_model])
    lb = LabelEncoder()
    y = lb.fit_transform(df[target_var])

    print(f"#Classes {len(lb.classes_)}")

    
    pca = PCA(n_components=N_PCS)
    X_pca =  pca.fit_transform(X)
    pca.explained_variance_ratio_.sum()

    results = np.zeros((N_PCS, N_PCS))

    for pc_1 in range(N_PCS):
        for pc_2 in range(pc_1 + 1, N_PCS):
            pcr = LogisticRegression()
            pcr.fit(X_pca[:, [pc_1, pc_2]], y)
            results[pc_1, pc_2] = pcr.score(X_pca[:, [pc_1, pc_2]], y) 
    
    return np.unravel_index(results.argmax(), shape=(N_PCS, N_PCS)), pca


def get_weighted_aggregated_embeddings_for_each_year(df: pd.DataFrame, embedding_column: str, aggregate_on: str):
    yearly_data = df.copy()
    yearly_data['year'] = pd.to_datetime(df['date']).dt.year
    groupped = yearly_data.groupby(by=[aggregate_on, 'year'])
    aggregated_embeddings = groupped[[embedding_column, 'migration_prob']].apply(lambda row: np.stack(row[embedding_column]).T @ np.stack(row['migration_prob']) / sum(row['migration_prob'])).reset_index()
    aggregated_embeddings.columns = [aggregate_on, 'year', embedding_column]
    return aggregated_embeddings



def get_aggregated_embeddings_for_each_year(df: pd.DataFrame, embedding_column: str, aggregate_on: str):
    yearly_data = df.copy()
    yearly_data['year'] = pd.to_datetime(df['date']).dt.year
    aggregated_embeddings = yearly_data.groupby(by=[aggregate_on, 'year'])[embedding_column].agg(lambda emb: np.stack(emb).mean(axis=0) )
    return aggregated_embeddings.reset_index()
