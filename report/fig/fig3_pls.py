import preamble


import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 

import preamble

from tueplots import bundles
from src.constants import COLOR_MAPS, PATH_VOCAB_EMBEDDED, EMBEDDING_MODEL, PATH_MIGRATION_SPEECHES_EMBEDDED
from src.dim_reduction import display_axis_semantics, closest_words_for_pc, get_extreme_examples, get_aggregated_embeddings_for_each_year

from sklearn.preprocessing import LabelEncoder
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import train_test_split

df = pd.read_parquet(PATH_MIGRATION_SPEECHES_EMBEDDED)

print(f"#Samples: {len(df)}")
vocab_df = pd.read_parquet(PATH_VOCAB_EMBEDDED)


aggregated = get_aggregated_embeddings_for_each_year(df, EMBEDDING_MODEL, 'block')


X = np.stack(df[EMBEDDING_MODEL])


lb = LabelEncoder()
y = lb.fit_transform(df["block"])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.1)
y = np.eye(len(lb.classes_))[y_train]  # create one-hot encodding to use PLS for classification PLS-DA


pls = PLSRegression(n_components=2)
pls.fit(X_train, y)

reduced = pls.transform(X)



def plot_aggregated_yearly_data(aggregated: pd.DataFrame, reduced_embeddings: np.array, target_var: str, color_map: dict, ax):
    ax.set_title("Two Dimensional Projection of Political Groups")

    grt_y = np.abs(reduced_embeddings[:, 1]).max() * 1.1
    grt_x = np.abs(reduced_embeddings[:, 0]).max() * 1.1

    ax.set_xlim(-grt_x, grt_x)
    ax.set_ylim(-grt_y, grt_y)

    scale = grt_y * 2

    unique_years = list(aggregated['year'].unique() )
    years_to_display = unique_years[::-2]

    ax.set_xlabel("First Axis")
    ax.set_ylabel("Second Axis")

    for party in aggregated[target_var].unique():
        party_mask = aggregated[target_var] == party
        years = aggregated[party_mask]['year']
        party_embeddings = reduced_embeddings[party_mask]
        
        ax.scatter(party_embeddings[:, 0], party_embeddings[:, 1], marker='o', color=color_map[party], label=party)
        for i, year in enumerate(years):
            if year in years_to_display:
                ax.text(party_embeddings[i,0] - scale * 0.025, party_embeddings[i,1]+ scale * 0.03, f"{year}",
                            fontsize=8, bbox=dict(boxstyle="round", color=color_map[party], alpha=0.7), 
                            color='white',
                            )

    ax.axhline(0, linestyle="--")
    ax.axvline(0, linestyle="--")
    ax.grid()
    ax.legend(loc="upper left")
    return ax


def plot_pca_axis_development(axis: int, aggregated: pd.DataFrame,  target_var: str, reduced_embeddings: np.array, 
                              axis_labels: tuple[list[str]], color_map: dict, ax: plt.Axes,
                              top_k: int = 3):

    ax.set_title(f"Development of political blocks over the years accross reduced-axis {axis + 1}")

    for party in aggregated[target_var].unique():
        party_mask = aggregated[target_var] == party
        years = aggregated[party_mask]['year']
        party_embeddings = reduced_embeddings[party_mask]
        ax.plot(years,  party_embeddings[:, axis], marker='o', color=color_map[party])

    
    max_y_lim = max(abs(ax.get_ylim()[0]), abs(ax.get_ylim()[1]))

    ax.set_ylim((-max_y_lim, max_y_lim))  
    ax.axhline(0, linestyle="--")

    ax_label_neg = ', '.join(map(lambda x: x[0], axis_labels[0][:top_k]))
    ax_label_pos = ', '.join(map(lambda x: x[0], axis_labels[1][:top_k]))

    props = dict(boxstyle='round', facecolor="grey", alpha=0.5)

    # label for negative axis: 
    ax.text(2014, -0.8 * max_y_lim , f"{ax_label_neg}", horizontalalignment="left", bbox=props)
    # label for positive axis: 
    ax.text(2014, 0.8 * max_y_lim, f"{ax_label_pos}", horizontalalignment="left", bbox=props)
    ax.grid()
    # ax.legend(loc="lower left")


def display_results(model, axis: tuple[int], aggregated: pd.DataFrame, vocab_df: pd.DataFrame,
                     reduced_embeddings: np.stack, target_var: str, color_map: dict) -> None:
    
    fig = plt.figure(layout="constrained", )

    gs0 = fig.add_gridspec(1, 2)

    gs1 = gs0[1].subgridspec(2, 1)

    ax1 = fig.add_subplot(gs0[0])
    ax2 = fig.add_subplot(gs1[0])
    ax3 = fig.add_subplot(gs1[1], sharex=ax2)

    axis_labels_0 = closest_words_for_pc(axis[0], model, vocab_df['word'], np.stack(vocab_df[EMBEDDING_MODEL]))
    axis_labels_1  = closest_words_for_pc(axis[1], model, vocab_df['word'], np.stack(vocab_df[EMBEDDING_MODEL]))


    plot_aggregated_yearly_data(aggregated, reduced_embeddings, target_var, color_map, ax1)
    plot_pca_axis_development(0, aggregated, target_var, reduced_embeddings, axis_labels_0, color_map, ax2)
    plot_pca_axis_development(1, aggregated, target_var, reduced_embeddings, axis_labels_1, color_map, ax3)
    return fig



params = bundles.icml2024(nrows=2,ncols=2, column="full") # if you need multiple columns / rows, change in your script
params.update({"figure.dpi": 350})
plt.rcParams.update(params)

fig = display_results(pls, [0, 1], aggregated, vocab_df, 
                pls.transform(np.stack(aggregated[EMBEDDING_MODEL])), "block", COLOR_MAPS['block'])

fig.savefig("report/fig/fig3.pdf")