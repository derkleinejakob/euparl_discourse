import sys
from pathlib import Path
import os 
from tueplots import bundles
from tueplots.constants.color import rgb
import matplotlib.pyplot as plt 

# add project root to sys path to be able to import ./src in notebooks that are in ./experiments
# Use __file__ if available (running as script), otherwise use cwd (running in notebook)
if '__file__' in globals():
    project_root = Path(__file__).resolve().parents[2]
else:
    project_root = Path.cwd().parents[1]

sys.path.append(str(project_root))

# change working directory to root so we can open data/file.csv from notebooks in ./experiments 
# and dont have to open ../data/file.csv
os.chdir(project_root)

# Hennigs version from plotting lecture:
params = bundles.icml2024() # if you need multiple columns / rows, change in your script
params.update({"figure.dpi": 350})
plt.rcParams.update(params)