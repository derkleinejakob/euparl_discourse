import preamble
import src.constants as const
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np#
from tueplots import bundles
import seaborn as sns
from scipy import stats

# import data
df = pd.read_parquet(const.PATH_MIGRATION_SPEECHES_SIMILARITIES)

selected_categories = [
    "immigrants_as_threat",
    "immigrants_as_problematic",
    "immigrants_humanitarian"
]

category_labels = {
    "immigrants_as_threat": "Supernarrative: \n \\textit{Immigration is a Threat}",
    "immigrants_as_problematic": "Supernarrative: \n \\textit{Immigrants' Culture is Problematic}",
    "immigrants_humanitarian": "Comparison narrative: \n \\textit{Humanitarian Principles in Migration}"
}

# filter df to only CHES years and compute correlations
ches_years = df['year_ches'].unique()
df_ches = df[df['year_x'].isin(ches_years)].copy()

# get CHES dimensions that were assessed in at least 2 different years
ches_dims = const.CHES_DIMENSIONS.copy()
for dim in const.CHES_DIMENSIONS:
    if len(df_ches['year_x'].loc[df_ches[dim].notna()].unique()) < 2:
        ches_dims.remove(dim)

# compute correlations for each narrative
topn_correlations = 3
alpha = 0.05
num_tests = len(selected_categories) * len(ches_dims)
corrected_alpha = alpha / num_tests

significance_data = {}
for ches_dim in ches_dims:
    for narrative in selected_categories:
        df_subset = df_ches.dropna(subset=[narrative, ches_dim])
        # if len(df_subset) > 10:
        cor, p_value = stats.pearsonr(df_subset[narrative], df_subset[ches_dim])
        significance_data[(ches_dim, narrative)] = (cor, p_value, p_value < corrected_alpha)
        # else:
        #     significance_data[(ches_dim, narrative)] = (np.nan, np.nan, False)

# get top 3 correlations for each narrative
top_ches_per_narrative = {}
for narrative in selected_categories:
    corrs = []
    for ches_dim in ches_dims:
        cor, p_value, is_significant = significance_data[(ches_dim, narrative)]
        if not pd.isna(cor) and is_significant:
            corrs.append((ches_dim, cor, p_value, is_significant))
    corrs.sort(key=lambda x: abs(x[1]), reverse=True)
    top_ches_per_narrative[narrative] = corrs[:topn_correlations]

# update plotting params for multiple subplots
params = bundles.icml2024(column = "full", nrows=2, ncols=3) 
params.update({"figure.dpi": 350})
plt.rcParams.update(params)

# build plot with 2 rows: line plots on top, heatmaps below
fig, axes = plt.subplots(2, 3, gridspec_kw={'hspace': 0})
# Top row
for i, category in enumerate(selected_categories):
    ax = axes[0, i]
    sns.lineplot(data=df, x='year_x', y=category, hue='block_x', markers=False, 
                 palette=const.COLOR_MAP_BLOCK, ax=ax, errorbar='ci', alpha=0.75)
    ax.set_title(category_labels[category])
    ax.set_xlabel("")
    ax.set_xlim(2014, 2024)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.2f}'.format(y)))
    if i == 0:
        ax.set_ylabel("Mean Cosine Similarity")
        handles, labels = ax.get_legend_handles_labels()
    else:
        ax.set_ylabel("")

    if category == "immigrants_humanitarian":
        ax.set_ylim(0.33, 0.459)
    ax.get_legend().remove()
    ax.set_axisbelow(True)
    ax.grid(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# Bottom row: mini heatmaps
# Add title for bottom row
fig.text(0, 0.23, 'Most \n Correla- \n ted Party \n Ratings', ha='left')

for i, category in enumerate(selected_categories):
    ax = axes[1, i]
    
    # Get top CHES dimensions for this narrative
    top_ches = top_ches_per_narrative[category]
    
    if len(top_ches) > 0:
        # Build mini correlation matrix (1 row, 3 columns) using absolute values
        ches_dim_names = [item[0] for item in top_ches]
        corr_values = np.array([[np.abs(item[1]) for item in top_ches]])  # Single row with absolute values
        corr_values = np.round(corr_values, 2)
        
        # Create heatmap
        sns.heatmap(corr_values, annot=False, cmap='Reds',
                    yticklabels=False, xticklabels=False, cbar=False,
                    ax=ax, vmin=0, vmax=1, linewidths=0.5, square=True)
        ax.set_ylabel("")
        ax.set_xlabel("")
        ax.set_xticks([])
        ax.set_yticks([])

        ches_labels = {"anti_islam_rhetoric": "Anti-Islam \n Rhetoric",
                       "eu_asylum": "EU-Asylum \n Policy",
                       "protectionism" : "Protectionism",
                       "people_vs_elite": "People \n vs. Elite",
                       "immigrate_salience" : "Migration \n Salience"
                       }
        
        # Add CHES dimension names and correlation values inside tiles
        # Add stars (*) for significant correlations
        for j, (ches_dim, cor, p_value, is_significant) in enumerate(top_ches):
            ax.text(j + 0.5, 0.35, ches_labels[ches_dim], ha='center', va='center', 
                   fontsize=7, weight='bold')
            star = '*' if is_significant else ''
            ax.text(j + 0.5, 0.70, f'{np.abs(cor):.2f}{star}', ha='center', va='center', 
                   fontsize=8)
    else:
        # Create blank heatmap with same dimensions as others (1 row, 3 columns)
        blank_corr = np.zeros((1, 3))
        sns.heatmap(blank_corr, annot=False, cmap='Reds',
                    yticklabels=False, xticklabels=False, cbar=False,
                    ax=ax, vmin=0, vmax=1, linewidths=0.5, square=True)
        ax.text(1.5, 0.5, 'No significant\ncorrelations', 
                ha='center', va='center', fontsize=7, weight='bold')
        ax.set_ylabel("")
        ax.set_xlabel("")
        ax.set_xticks([])
        ax.set_yticks([])

# Manually adjust subplot positions to reduce vertical spacing
for i in range(3):
    # Get positions
    pos_top = axes[0, i].get_position()
    pos_bottom = axes[1, i].get_position()
    
    # Move bottom row up closer to top row
    new_bottom = pos_top.y0 - 0.3  # Adjust this value to control spacing
    axes[1, i].set_position([pos_bottom.x0, new_bottom, pos_bottom.width, pos_bottom.height])

fig.legend(handles=handles, labels=[const.LEGEND_BLOCK[label] for label in labels], 
           loc='upper center', ncol=len(const.ORDER_BLOCK), frameon=True, 
           bbox_to_anchor=(0.5, 0.92), fancybox=True, shadow=False)

fig.savefig("report/fig/fig4_search.pdf", bbox_inches='tight')