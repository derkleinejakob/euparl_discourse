import preamble
import pandas as pd 
import matplotlib.pyplot as plt
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

# create stacked area plot
ax1 = df_pivot.plot.area(cmap="viridis", alpha=0.75)
ax1.set_ylabel('Proportion of Speeches')
ax1.set_xlabel('')

# reverse legend order
handles, labels = ax1.get_legend_handles_labels()
ax1.legend(handles[::-1], labels[::-1], loc='upper left', frameon=True, fancybox=True, shadow=False)
# ax1.set_title('Proportion of Speeches by Dominant Topic Over Time')
ax1.set_xlim(df_dominant['year'].min(), df_dominant['year'].max())
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'.replace('%', r'\%')))
ax1.set_axisbelow(True)

ax1.grid(alpha=0.3)
# despine
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# save figure in case we use only this panel
fig1 = ax1.get_figure()
fig1.savefig("report/fig/fig1_panel1.pdf")

# ----- panel 2: number of migration speeches per party block over time ---------------------------------

# create migration only dataframe
df_migration = df[df[f'topic_{const.MIGRATION_TOPIC_ID}'] >= const.MIGRATION_THRESHOLD]

# aggregate number of speeches per block and year
df_migration = df_migration.groupby(['year', 'block']).agg({
    'text': 'count'}).reset_index().rename(columns={'text': 'n_speeches'})

# calculate ratio
df_migration['total_speeches'] = df_migration['year'].map(total_speeches_per_year)
df_migration['ratio'] = df_migration['n_speeches'] / df_migration['total_speeches']

# make sure blocks are categorical and in correct order
df_migration['block'] = pd.Categorical(df_migration['block'], categories=const.ORDER_BLOCK, ordered=True)
df_migration = df_migration.sort_values(['year', 'block'])

ax2 = df_migration.pivot(index='year', columns='block', values='ratio').plot.area(
    stacked=True, 
    color=[const.COLOR_MAP_BLOCK[block] for block in const.ORDER_BLOCK],
    alpha=0.75
)

ax2.set_xlabel("")
ax2.set_ylabel('Proportion of Speeches per Year')
# ax.set_title("Migration Discourse by Political Block")

# reverse legend order
handles, labels = ax2.get_legend_handles_labels()
# rename labels using dictionary
labels = [const.LEGEND_BLOCK[label] for label in labels]

ax2.legend(handles[::-1], labels[::-1], loc='upper right', frameon=True)

# grid
ax2.set_axisbelow(True)
ax2.grid(alpha=0.3)

# ax.vlines(x=const.ELECTION_YEARS, ymin=0, ymax=1, colors='gray', linestyles='dashed', alpha=0.5)

ax2.set_xlim(2014, 2024)
ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'.replace('%', r'\%')))

# despine
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

# save figure in case we use only this panel
fig2 = ax2.get_figure()
fig2.savefig("report/fig/fig1_panel2.pdf")

# ------- combine panels into one figure ------------------------------------------------------
params = bundles.icml2024(nrows=2,ncols=1) 
params.update({"figure.dpi": 350})
plt.rcParams.update(params)

fig, (ax1_combined, ax2_combined) = plt.subplots(2, 1)
# panel 1
df_pivot.plot.area(cmap="viridis", alpha=0.75, ax=ax1_combined)
ax1_combined.set_ylabel('Proportion of Speeches per Year')
ax1_combined.set_xlabel('')
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
df_migration.pivot(index='year', columns='block', values='n_speeches').plot.area(
    stacked=True, 
    color=[const.COLOR_MAP_BLOCK[block] for block in const.ORDER_BLOCK],
    alpha=0.75,
    ax=ax2_combined
)
ax2_combined.set_xlabel("")
ax2_combined.set_ylabel('Number of Migration Speeches')
handles, labels = ax2_combined.get_legend_handles_labels()
labels = [const.LEGEND_BLOCK_LONG[label] for label in labels]
ax2_combined.legend(handles[::-1], labels[::-1], loc='upper right', frameon=True)
ax2_combined.set_axisbelow(True)
ax2_combined.grid(alpha=0.3)
ax2_combined.set_xlim(2014, 2024)
ax2_combined.spines['top'].set_visible(False)
ax2_combined.spines['right'].set_visible(False)

# add arrow annotation
# ax1_combined.annotate(
#     '', 
#     xy=(0.55, 0.50), xycoords='figure fraction', 
#     xytext=(0.55, 0.58), textcoords='figure fraction',
#     arrowprops=dict(color='#440154', lw=1)
# )

# fig.tight_layout()
fig.savefig("report/fig/fig1_combined.pdf")


