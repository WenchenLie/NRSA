import json
from math import pi, sqrt, isclose
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


root = Path(r'H:\SDOFresults_test')


job = json.load(open(root / 'job.json'))
T = job['Periods']
gm_names = job['Ground motion names']
all_R = np.zeros((len(T), len(gm_names)))
for i, gm_name in enumerate(gm_names):
    df = pd.read_csv(root / f'results/{gm_name}.csv')
    R = df['R']
    all_R[:, i] = R
    plt.plot(T, R, color='gray')
mean_R = np.mean(all_R, axis=1)
plt.plot(T, mean_R, color='red')
plt.show()

