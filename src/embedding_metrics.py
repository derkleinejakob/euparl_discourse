from sklearn.preprocessing import LabelEncoder, KBinsDiscretizer
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import homogeneity_score, completeness_score, v_measure_score
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_score
from sklearn.cluster import KMeans

import pandas as pd
import numpy as np

def get_intra_inter_similarities(df: pd.DataFrame, model: str, group_col: str, weighted: bool = False) -> list[dict]:
    embeddings = np.stack(df[model])
    normalized_embeddings = embeddings / np.linalg.vector_norm(embeddings, axis=1, ord=2, keepdims=True)
    similarities = normalized_embeddings @ normalized_embeddings.T

    # for computing weighted similarities instead of raw.
    # we want to weight them by IDA's topic probability, lower prob migration speeches might belong to other topics 
    # or might be too different and we don't want to let outliers skew the results
    if weighted:
        mig_prob_vec = np.stack(df['migration_prob'])
        mig_prob_vec = mig_prob_vec / np.linalg.vector_norm(mig_prob_vec, ord=2)
        weights = mig_prob_vec[:, np.newaxis] @ mig_prob_vec[np.newaxis, :]
        similarities *= weights

    classes = df[group_col].unique()
    results = []

    for class_ in classes:
        intra = similarities[df[group_col] == class_][:, df[group_col] == class_].mean()
        inter = similarities[df[group_col] == class_][:, df[group_col] != class_].mean()
        results.append({'class': class_, 'intra': intra, 'inter': inter, 'size': sum(df[group_col] == class_)})
    return results


def get_cohesiveness(df: pd.DataFrame, model: str, group_col: str, weighted: bool = False) -> float:
    similarities = get_intra_inter_similarities(df=df, model=model, group_col=group_col, weighted=weighted)
    size = sum(sim['size'] for sim in similarities)
    cohesivness = sum((sim['intra'] / sim['inter'] - 1) * sim['size'] for sim in similarities) / size
    return cohesivness.item()





def evaluate_kmeans(true_labels, cluster_labels) -> dict[str, float]:
    h = homogeneity_score(true_labels, cluster_labels)
    c = completeness_score(true_labels, cluster_labels)
    v = v_measure_score(true_labels, cluster_labels)
 
    return {"homogeneity": h, "completeness": c, "v_measure": v}


def get_cluster_quality(df: pd.DataFrame, model: str, target_var: str, weighted: bool = False) -> dict[str, float]:
    kmeans = KMeans(n_clusters=len(df[target_var].unique()), random_state=42)
    predicted_clusters = kmeans.fit_predict(np.stack(df[model]), sample_weight=df['migration_prob'] if weighted else None)
    return evaluate_kmeans(true_labels=df[target_var], cluster_labels=predicted_clusters)




def pls_coefficient_of_determination(df: pd.DataFrame, model: str, target_var: str, categorical: bool = True) -> float:
    X = np.stack(df[model])
    if categorical:
        # If we have categorical target, create one-hot encodding to adapt PLS for classification (PLS-DA)
        lb = LabelEncoder()
        y = lb.fit_transform(df[target_var])

        y = np.eye(len(lb.classes_))[y]
    else:
        y = np.stack(df[target_var])

    X_train, X_test, y_train, y_test = train_test_split(X, y)
    pls = PLSRegression(n_components=2)
    pls.fit(X_train, y_train)

    return pls.score(X_test, y_test)



def compute_predictive_power(df: pd.DataFrame, emb_model: str, target_var: str, continues: bool = False) -> np.array:
    print(f"Predicting {target_var}")

    model = LogisticRegression(max_iter=1_000)
    X = np.stack(df[emb_model])
    y = df[target_var]

    print(f"#Classes {len(y.unique())}")

    if continues:
        y = KBinsDiscretizer(encode="ordinal").fit_transform(np.stack(y)[:, np.newaxis]).reshape(-1)

    y = LabelEncoder().fit_transform(y)

    cv = StratifiedKFold(n_splits=10, shuffle=True)
    scores = cross_val_score(model, X, y, cv=cv, scoring="f1_macro")
    print(f"Mean Macro F1: {scores.mean()}")
    print(f"STD Macro F1: {scores.std()}")

    return scores