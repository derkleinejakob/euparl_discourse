import sys
from pathlib import Path
import os 
from tueplots import bundles
from tueplots.constants.color import rgb
import matplotlib.pyplot as plt 

if __name__ == "__main__":
    # to allow sub-folders in ./experiments, added command line argument (only one argument, assume integer, default 1)
    # => if notebook is located in sub-folder, can increase number of steps-up the script should use to find the actual 
    # project root.
    n_parents_up = sys.argv[1] if len(sys.argv) > 1 else 1 
    # add project root to sys path to be able to import ./src in notebooks that are in ./experiments
    project_root = Path.cwd()
    for _ in range(int(n_parents_up)):
        project_root = project_root.parent
        
    sys.path.append(str(project_root))
    sys.path.append(str(project_root)+"/src")

    # change working directory to root so we can open data/file.csv from notebooks in ./experiments 
    # and dont have to open ../data/file.csv
    os.chdir(project_root)

    # Hennigs version from plotting lecture:
    params = bundles.icml2024() # if you need multiple columns / rows, change in your script
    params.update({"figure.dpi": 350})
    plt.rcParams.update(params)

    os.environ["PREAMBLE_RUN"] = "True"