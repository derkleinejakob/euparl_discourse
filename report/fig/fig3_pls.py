import preamble


import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 

import preamble

from tueplots import bundles
from src.constants import COLOR_MAPS, PATH_VOCAB_EMBEDDED, EMBEDDING_MODEL, PATH_MIGRATION_SPEECHES_EMBEDDED, LEGEND_BLOCK, ORDER_BLOCK
from src.dim_reduction import display_axis_semantics, closest_words_for_pc, get_aggregated_embeddings_for_each_year

from sklearn.preprocessing import LabelEncoder
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import train_test_split

df = pd.read_parquet(PATH_MIGRATION_SPEECHES_EMBEDDED)

np.random.seed(0)

df = df[~df['written']]

print(f"#Samples: {len(df)}")
vocab_df = pd.read_parquet(PATH_VOCAB_EMBEDDED)

df['block'] = pd.Categorical(df['block'], categories=ORDER_BLOCK, ordered=True)


aggregated = get_aggregated_embeddings_for_each_year(df, EMBEDDING_MODEL, 'block')


X = np.stack(df[EMBEDDING_MODEL])

vocab_df['word'] = vocab_df['word'].str.replace("humanrights", "human-rights-")  # small correction for visual aesthetics


lb = LabelEncoder()
y = lb.fit_transform(df["block"])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.1)
y = np.eye(len(lb.classes_))[y_train]  # create one-hot encodding to use PLS for classification PLS-DA


pls = PLSRegression(n_components=2)
pls.fit(X_train, y)

reduced = pls.transform(X)



import seaborn as sns
def plot_pca_axis_development(df: pd.DataFrame, axis: int,  target_var: str,
                              axis_labels: tuple[list[str]], color_map: dict, ax: plt.Axes,
                              top_k: int = 3):

    # ax.set_title(f"PLS Axis: {axis + 1}")


    sns.lineplot(data=df, x='year', y=f"reduced_{axis}",  hue=target_var, palette=color_map, ax=ax, legend=False,
                  err_kws={"alpha": 0.08}, linewidth=1.8)
    ax.set_xlabel("")
    ax.set_ylabel(f"PLS {axis + 1}")
    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)
    ax.tick_params(top=False, right=False)

    
    max_y_lim = max(abs(ax.get_ylim()[0]), abs(ax.get_ylim()[1]))

    ax.set_ylim((-max_y_lim, max_y_lim))  
    ax.set_xlim(2014, 2024)
    ax.axhline(0, linestyle="--" , lw=1.0, alpha=.5)

    ax_label_neg = ', '.join(map(lambda x: x[0], axis_labels[0][:top_k]))
    ax_label_pos = ', '.join(map(lambda x: x[0], axis_labels[1][:top_k]))

    props = dict(boxstyle='round', facecolor="white", alpha=0.85,  edgecolor="0.7",)

    # label for negative axis: 
    ax.text(0.02, 0.08,  f"{ax_label_neg}",  transform=ax.transAxes, va="bottom", ha="left", horizontalalignment="left", bbox=props)
    # label for positive axis: 
    ax.text(0.02, 0.9, f"{ax_label_pos}", transform=ax.transAxes, va="top", ha="left",  horizontalalignment="left", bbox=props)
    ax.grid()
    # ax.legend(loc="lower left")


def display_results(df: pd.DataFrame, model, axis: tuple[int], aggregated: pd.DataFrame, vocab_df: pd.DataFrame,
                     reduced_embeddings: np.stack, target_var: str, color_map: dict) -> None:
    
    fig = plt.figure()

    gs1 = fig.add_gridspec(2, 1)

    ax2 = fig.add_subplot(gs1[0])
    ax3 = fig.add_subplot(gs1[1], sharex=ax2)

    ax2.tick_params(labelbottom=False)

    

    axis_labels_0 = closest_words_for_pc(axis[0], model, vocab_df['word'], np.stack(vocab_df[EMBEDDING_MODEL]))
    axis_labels_1  = closest_words_for_pc(axis[1], model, vocab_df['word'], np.stack(vocab_df[EMBEDDING_MODEL]))

    display_axis_semantics([(axis_labels_0), 
                            (axis_labels_1)])

    # plot_aggregated_yearly_data(aggregated, reduced_embeddings, target_var, color_map, ax1)
    plot_pca_axis_development(df, 0, target_var,  axis_labels_0, color_map, ax2)
    plot_pca_axis_development(df, 1, target_var, axis_labels_1, color_map, ax3)

    # handles, labels = ax2.get_legend_handles_labels()
    # labels = [LEGEND_BLOCK[label] for label in labels]
    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D([0], [0], color=color_map[k], lw=2, label=LEGEND_BLOCK[k])
        for k in color_map
    ]
    
    fig.legend(
        handles=legend_elements,
        loc="lower center",
        ncol=len(legend_elements) // 2,
        frameon=True,
        bbox_to_anchor=(0.5, -0.12)
    )
    # fig.subplots_adjust(bottom=0.22)

    
    return fig




params = bundles.icml2024(nrows=2, ncols=1) # if you need multiple columns / rows, change in your script
params.update({"figure.dpi": 350})
plt.rcParams.update(params)

df['reduced_0'] = reduced[:, 0] 
df['reduced_1'] = reduced[:, 1] 

fig = display_results(df, pls, [0, 1], aggregated, vocab_df, 
                pls.transform(np.stack(aggregated[EMBEDDING_MODEL])), "block", COLOR_MAPS['block'])

fig.savefig("report/fig/fig3.pdf", bbox_inches="tight")
fig.savefig("report/fig/fig3.png", bbox_inches="tight")