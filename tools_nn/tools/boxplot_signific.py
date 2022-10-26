import seaborn as sns
import matplotlib.pyplot as plt
from statannotations.Annotator import Annotator
import pandas as pd
import numpy as np

path_nnunet = '/mnt/CRAI-NAS/all/martinsr/NNunet/report/validation/stats_2022-10-23_val_NNUnet.csv'
path_bayesian = '/mnt/CRAI-NAS/all/martinsr/NNunet/report/brno/stats_2022-10-23_brno_bayesian.csv'
path_2DUnet = '/mnt/CRAI-NAS/all/martinsr/NNunet/report/validation/stats_2022-10-25_val_25DUNet.csv'

df = pd.read_csv(path_nnunet)
# make a copy of df
df2 = pd.read_csv(path_bayesian)
# add together and change all the model values
df3 = pd.read_csv(path_2DUnet)
# stack into df
# df = pd.concat([df, df2, df3], axis=0)
df = pd.concat([df, df3], axis=0)

# locate rows with nan in h95
df_nan = df[df['h95'].isna()]
participants_nan = df_nan['participant']
# set h95 values for participants to nan for all models
df.loc[df['participant'].isin(participants_nan), 'h95'] = np.nan


import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def subplots_centered(nrows, ncols, figsize, nfigs):
    """
    Modification of matplotlib plt.subplots(),
    useful when some subplots are empty.
    
    It returns a grid where the plots
    in the **last** row are centered.
    
    Inputs
    ------
        nrows, ncols, figsize: same as plt.subplots()
        nfigs: real number of figures
    """
    assert nfigs < nrows * ncols, "No empty subplots, use normal plt.subplots() instead"
    
    fig = plt.figure(figsize=figsize)
    axs = []
    
    m = nfigs % ncols
    m = range(1, ncols+1)[-m]  # subdivision of columns
    gs = gridspec.GridSpec(nrows, m*ncols)

    for i in range(0, nfigs):
        row = i // ncols
        col = i % ncols

        if row == nrows-1: # center only last row
            off = int(m * (ncols - nfigs % ncols) / 2)
        else:
            off = 0

        ax = plt.subplot(gs[row, m*col + off : m*(col+1) + off])
        axs.append(ax)
        
    return fig, axs

fig, ax = subplots_centered(nrows=2, ncols=3, figsize=(6,10), nfigs=5)

# fig, ax = plt.subplots(2, 3 , figsize=(6, 10))
# fig.subplots_adjust(hspace=0.5, wspace=0.5)
# set the title of the figure
sns.boxplot(x="model", y="dsc", data=df, ax=ax[0])
sns.boxplot(x="model", y="h95", data=df, ax=ax[1])
sns.boxplot(x="model", y="avd", data=df, ax=ax[2])
sns.boxplot(x="model", y="recall", data=df, ax=ax[3])
sns.boxplot(x="model", y="f1", data=df, ax=ax[4])

# flip the x axis labels
for i in range(5):
    ax[i].set_xticklabels(ax[i].get_xticklabels(), rotation=90)
# plt.tight_layout()
# plt.savefig('boxplot.png')

# pairs = [("Deep Bayesian", "3D nnU-Net"), ("Deep Bayesian", "2.5DUNet"), ("3D nnU-Net", "2.5DUNet")]
pairs = [("3D nnU-Net", "25DUNet")]
test = "Wilcoxon"
correction = "Bonferroni"
# custom_pvalues = [(0.1, 0.2), (0.1, 0.2), (0.1, 0.2)]
custom_pvalues = [0.1]
# add the statistical annotations
annotator = Annotator(ax[0], data=df, x="model", y="dsc", pairs = pairs)
annotator.configure(text_format='star', loc='outside')
# annotator.set_pvalues(pvalues=custom_pvalues)
# annotator.configure(test=None, text_format='star', loc='outside', verbose=2)
# annotator.apply_and_annotate()
annotator.set_pvalues_and_annotate(pvalues=custom_pvalues)
annotator.configure(text_format='star', loc='outside')

annotator = Annotator(ax[1], data=df, x="model", y="h95", pairs = pairs)
# annotator.configure(comparisons_correction=correction, test=None, text_format='star', loc='outside')
# annotator.set_pvalues(pvalues=custom_pvalues)
annotator.set_pvalues_and_annotate(pvalues=custom_pvalues)
annotator.configure(text_format='star', loc='outside')

annotator = Annotator(ax[2], data=df, x="model", y="avd", pairs = pairs)
# annotator.configure(comparisons_correction=correction, test=None, text_format='star', loc='outside')
# annotator.set_pvalues(pvalues=custom_pvalues)
annotator.set_pvalues_and_annotate(pvalues=custom_pvalues)
annotator.configure(text_format='star', loc='outside')

annotator = Annotator(ax[3], data=df, x="model", y="recall", pairs = pairs)
# annotator.configure(comparisons_correction=correction, test=None, text_format='star', loc='outside')
# annotator.set_pvalues(pvalues=custom_pvalues)
annotator.set_pvalues_and_annotate(pvalues=custom_pvalues)
annotator.configure(text_format='star', loc='outside')

annotator = Annotator(ax[4], data=df, x="model", y="f1", pairs = pairs)
# annotator.configure(comparisons_correction=correction, test=None, text_format='star', loc='outside')
# annotator.set_pvalues(pvalues=custom_pvalues)
annotator.set_pvalues_and_annotate(pvalues=custom_pvalues)
annotator.configure(text_format='star', loc='outside')


# make the figure tight
plt.tight_layout()
# save the figure
plt.savefig('plots_and_metrics_wmh/boxplot.jpg')
plt.savefig('plots_and_metrics_wmh/boxplot.svg')
plt.savefig('plots_and_metrics_wmh/boxplot.tif')

