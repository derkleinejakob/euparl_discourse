import preamble
import pandas as pd 
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from gensim.models import LdaModel
import src.constants as const
from src.print_topics import print_topics
from tueplots import bundles
from tueplots.constants.color import rgb
import src.constants as const  

## TODO: subtitles like "selected topics over time" and "migration discourse over time"

# import data
df = pd.read_parquet(const.PATH_ALL_SPEECHES)
total_speeches_per_year = df.groupby('year').size().to_dict() # for ratios

# --- panel 1: number of speeches assigned to several selected topics plotted over time ---------------------------------

# select topic with highest probability for each speech
df['dominant_topic_id'] = df.loc[:, 'topic_0':'topic_29'].idxmax(axis=1).apply(lambda x: int(x.split('_')[1])).astype(int)
# give dominant topics their label
df['dominant_topic'] = df['dominant_topic_id'].apply(lambda x: const.TOPIC_LABELS[x])

# aggregate number of speeches per dominant topic and year
df_dominant = df.groupby(['dominant_topic', 'year']).agg({
    'text': 'count'}).reset_index().rename(columns={'text': 'count'})

# calculate ratios
df_dominant['total_speeches'] = df_dominant['year'].map(total_speeches_per_year)
df_dominant['ratio'] = df_dominant['count'] / df_dominant['total_speeches']

chosen_topics = [
    "Migration / Asylum",
    "Terror / Political Violence",
    "Climate / Energy",
    "Disasters / Epidemics",
    "Russia / Ukraine"]

df_chosen = df_dominant[df_dominant['dominant_topic'].isin(chosen_topics)]

# pivot dataframe for stacked area plot
df_pivot = df_chosen.pivot(index='year', columns='dominant_topic', values='ratio').fillna(0)

# sort columns in chosen_topics order
df_pivot = df_pivot[chosen_topics]

# ----- panel 2: number of migration speeches per party block over time ---------------------------------

# create migration only dataframe
df_migration = df[df[f'topic_{const.MIGRATION_TOPIC_ID}'] >= const.MIGRATION_THRESHOLD]

# aggregate number of speeches per block, year, and written status
df_migration = df_migration.groupby(['year', 'block', 'written']).agg({
    'text': 'count'}).reset_index().rename(columns={'text': 'n_speeches'})

# calculate ratio
df_migration['total_speeches'] = df_migration['year'].map(total_speeches_per_year)
df_migration['ratio'] = df_migration['n_speeches'] / df_migration['total_speeches']

# make sure blocks are categorical and in correct order
df_migration['block'] = pd.Categorical(df_migration['block'], categories=const.ORDER_BLOCK, ordered=True)
df_migration = df_migration.sort_values(['year', 'block', 'written'])

def plot_migration_by_block_written(df_in, value_col, ax, legend_labels, legend_loc="upper right"):
    pivot = df_in.pivot(index='year', columns=['block', 'written'], values=value_col).fillna(0)
    col_order = []
    for block in const.ORDER_BLOCK:
        for written in [False, True]:
            if (block, written) in pivot.columns:
                col_order.append((block, written))
    pivot = pivot[col_order]

    colors = [const.COLOR_MAP_BLOCK[block] for block, written in pivot.columns]
    pivot.plot.area(stacked=True, color=colors, alpha=0.75, ax=ax)

    for (block, written), collection in zip(pivot.columns, ax.collections):
        base_color = const.COLOR_MAP_BLOCK[block]
        face_rgba = mcolors.to_rgba(base_color, alpha=1)
        collection.set_alpha(1)
        collection.set_facecolor(face_rgba)
        collection.set_edgecolor((1, 1, 1, 1))
        # collection.set_hatch('////')
        # collection.set_linewidth(0.8)

        if written:
            collection.set_hatch('////')
            collection.set_linewidth(0)
        else:
            collection.set_hatch(None)

    blocks_present = [block for block in const.ORDER_BLOCK if any((block, w) in pivot.columns for w in [False, True])]
    block_handles = [mpatches.Patch(facecolor=const.COLOR_MAP_BLOCK[block], label=legend_labels[block]) for block in blocks_present]
    written_handle = mpatches.Patch(facecolor='white', edgecolor='black', label='Written Speeches', hatch='////')
    legend_handles = block_handles[::-1] + [written_handle]
    ax.legend(handles=legend_handles, loc=legend_loc, frameon=True)

    return pivot

# ------- build figure --------------------------------------------

params = bundles.icml2024(nrows=2,ncols=1) 
params.update({"figure.dpi": 350, "figure.figsize": (params["figure.figsize"][0], 3.2)})
plt.rcParams.update(params)

fig, (ax1_combined, ax2_combined) = plt.subplots(2, 1)
# panel 1
df_pivot.plot.area(cmap="viridis", alpha=0.75, ax=ax1_combined)
ax1_combined.set_ylabel('Proportion of Speeches')
ax1_combined.set_xlabel('')
ax1_combined.set_xticklabels([])

# ax1_combined.set_xticklabels([])
handles, labels = ax1_combined.get_legend_handles_labels()
ax1_combined.legend(handles[::-1], labels[::-1], loc='upper left', frameon=True, fancybox=True, shadow=False)
ax1_combined.set_xlim(df_dominant['year'].min(), df_dominant['year'].max())
ax1_combined.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'.replace('%', r'\%')))
ax1_combined.set_axisbelow(True)
ax1_combined.grid(alpha=0.3)
ax1_combined.spines['top'].set_visible(False)
ax1_combined.spines['right'].set_visible(False)
# panel 2
plot_migration_by_block_written(df_migration, 'n_speeches', ax2_combined, const.LEGEND_BLOCK_LONG)
ax2_combined.set_xlabel("")
ax2_combined.set_ylabel('Number of Migration Speeches')
ax2_combined.set_axisbelow(True)
ax2_combined.grid(alpha=0.3)
ax2_combined.set_xlim(2014, 2024)
ax2_combined.spines['top'].set_visible(False)
ax2_combined.spines['right'].set_visible(False)

ax1_combined.yaxis.set_label_coords(-0.12, 0.5)
ax2_combined.yaxis.set_label_coords(-0.12, 0.5)


# add arrow annotation
# ax1_combined.annotate(
#     '', 
#     xy=(0.55, 0.50), xycoords='figure fraction', 
#     xytext=(0.55, 0.58), textcoords='figure fraction',
#     arrowprops=dict(color='#440154', lw=1)
# )

# fig.tight_layout()
fig.savefig("report/fig/fig1_lda.pdf")

