import sys
from pathlib import Path
import os 
from tueplots import bundles
from tueplots.constants.color import rgb
import matplotlib.pyplot as plt 

# add project root to sys path to be able to import ./src in notebooks that are in ./experiments
project_root = Path.cwd().parent
sys.path.append(str(project_root))

# change working directory to root so we can open data/file.csv from notebooks in ./experiments 
# and dont have to open ../data/file.csv
os.chdir(project_root)

# Hennigs version from plotting lecture:
params = bundles.icml2024() # if you need multiple columns / rows, change in your script
params.update({"figure.dpi": 350})
plt.rcParams.update(params)
# jakobs version:
# inspired by Prof. Hennig's preamble: set matplotlib's default to Uni Tübingen's params
# plt.rcParams.update(bundles.beamer_moml())
# plt.rcParams.update({"figure.dpi": 350})