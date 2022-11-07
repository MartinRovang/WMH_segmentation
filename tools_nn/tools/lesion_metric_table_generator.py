from turtle import color
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

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

path = "/mnt/CRAI-NAS/all/martinsr/NNunet/tools/plots_and_metrics_wmh/lesion_metrics/test/lesion_metrics_25D_test.csv"
pathfp = "/mnt/CRAI-NAS/all/martinsr/NNunet/tools/plots_and_metrics_wmh/lesion_metrics/test/lesion_metrics_fp_25D_test.csv"
df25 = pd.read_csv(path)
dffp25 = pd.read_csv(pathfp)

path = "/mnt/CRAI-NAS/all/martinsr/NNunet/tools/plots_and_metrics_wmh/lesion_metrics/test/lesion_metrics_bayesian_test.csv"
pathfp = "/mnt/CRAI-NAS/all/martinsr/NNunet/tools/plots_and_metrics_wmh/lesion_metrics/test/lesion_metrics_fp_bayesian_test.csv"
dfbay = pd.read_csv(path)
dffpbay = pd.read_csv(pathfp)

path = "/mnt/CRAI-NAS/all/martinsr/NNunet/tools/plots_and_metrics_wmh/lesion_metrics/test_fixed_threshold/lesion_metrics_3DNNunet_test.csv"
pathfp = "/mnt/CRAI-NAS/all/martinsr/NNunet/tools/plots_and_metrics_wmh/lesion_metrics/test_fixed_threshold/lesion_metrics_fp_3DNNunet_test.csv"
dfnn = pd.read_csv(path)
dffpnn = pd.read_csv(pathfp)

# # add column for model type
# df25['Model'] = ['2.5D nnU-Net']*len(df25)
# dffp25['Model'] = ['2.5D nnU-Net']*len(dffp25)
# dfbay['Model'] = ['Deep Bayesian']*len(dfbay)
# dffpbay['Model'] = ['Deep Bayesian']*len(dffpbay)
# dfnn['Model'] = ['nnU-Net']*len(dfnn)
# dffpnn['Model'] = ['nnU-Net']*len(dffpnn)


# interval >= 0 and < 0.01

def lesion_brackets(df, dfp, modelname):
    df['Model'] = [modelname]*len(df)
    dfp['Model'] = [modelname]*len(dfp)
    level1 = df[(df['Lesion Size'] > 0) & (df['Lesion Size'] < 0.01)] 
    level2 = df[(df['Lesion Size'] >= 0.01) & (df['Lesion Size'] < 0.4)]
    level3 = df[(df['Lesion Size'] >= 0.4) & (df['Lesion Size'] < 1)]
    level4 = df[df['Lesion Size'] >= 1]

    level1fp = dfp[(dfp['Lesion Size'] > 0) & (dfp['Lesion Size'] < 0.01)]
    level2fp = dfp[(dfp['Lesion Size'] >= 0.01) & (dfp['Lesion Size'] < 0.4)]
    level3fp = dfp[(dfp['Lesion Size'] >= 0.4) & (dfp['Lesion Size'] < 1)]
    level4fp = dfp[dfp['Lesion Size'] >= 1]

    df["Lesion Sizes [mL]"] = pd.cut(df["Lesion Size"], bins=[0, 0.2, 0.4, 1, 100], labels=["(0,0.2]", "(0.2,0.4]", "(0.4,1]", "(1,100]"])
    dfp["Lesion Sizes [mL]"] = pd.cut(dfp["Lesion Size"], bins=[0, 0.2, 0.4, 1, 100], labels=["(0,0.2]", "(0.2,0.4]", "(0.4,1]", "(1,100]"])
    # df["Lesion Sizes [mL]"] = pd.cut(df["Lesion Size"], bins=[0, 0.2, 0.4, 1, 100], labels=["(0,0.2]", "(0.2,0.4]", "(0.4,1]", "(1-100]"])
    # dfp["Lesion Sizes [mL]"] = pd.cut(dfp["Lesion Size"], bins=[0, 0.2, 0.4, 1, 100], labels=["(0,0.2]", "(0.2,0.4]", "(0.4-1]", "(1-100]"])

    return level1, level2, level3, level4, level1fp, level2fp, level3fp, level4fp, df, dfp

level1, level2, level3, level4, level1fp, level2fp, level3fp, level4fp, df, dfp = lesion_brackets(df25, dffp25, '2.5D U-Net')
level1b, level2b, level3b, level4b, level1fpb, level2fpb, level3fpb, level4fpb, dfb, dfpb = lesion_brackets(dfbay, dffpbay, 'Deep Bayesian')
level1n, level2n, level3n, level4n, level1fpn, level2fpn, level3fpn, level4fpn, dfn, dfpn = lesion_brackets(dfnn, dffpnn, '3D nnU-Net')

# col1 = f"""{round(np.mean(level1['Lesion Size']),3)} |  {round(np.mean(level2['Lesion Size']),3)} | {round(np.mean(level3['Lesion Size']),3)} | {round(np.mean(level4['Lesion Size']),3)}"""
# col2 = f"""{len(level1fp)} |  {len(level2fp)} | {len(level3fp)} | {len(level4fp)}"""
# col3 = f""" {np.sum(level1["Recall"] >= 0.5)} | {np.sum(level2["Recall"] >= 0.5)} | {np.sum(level3["Recall"] >= 0.5)} | {np.sum(level4["Recall"] >= 0.5)}"""
# col4 = f""" {np.sum([level1["Recall"] < 0.5])} | {np.sum([level2["Recall"] < 0.5])} | {np.sum([level3["Recall"] < 0.5])} | {np.sum([level4["Recall"] < 0.5])}"""
# col5 = f"""{round(np.mean(level1fp["Precision"]),3)} | {round(np.mean(level2fp["Precision"]),3)} | {round(np.mean(level3fp["Precision"]),3)} | {round(np.mean(level4fp["Precision"]),3)}"""
# col6 = f"""{round(np.mean(level1['Recall']),3)} |  {round(np.mean(level2['Recall']), 3)} | {round(np.mean(level3['Recall']), 3)} | {round(np.mean(level4['Recall']), 3)}"""

# print(col1)
# print(col2)
# print(col3)
# print(col4)
# print(col5)
# print(col6)



sns.set_theme(style="whitegrid")
color_list = sns.color_palette()
color_dark = sns.color_palette("dark")
# df = pd.concat([df25, dfbay, dfnn])
dfmain = pd.concat([df, dfb, dfn], ignore_index=True)

# add gt lesions
dfmainfp = pd.concat([dffp25, dfpb, dffpnn], ignore_index=True)

# same figure
f, ax = plt.subplots(3,1, figsize=(10, 10), sharex=False)

df__ = pd.DataFrame(df[df["Model"] == "2.5D U-Net"])
# change model name to Ground truth
df__.loc[df__.index, 'Model'] = "Ground truth"

# add to dfmainfp
dfmainfp = pd.concat([dfmainfp, df__], ignore_index=True)

sns.despine(bottom=True, left=True)
# Show each observation with a scatterplot # set colors
sns.stripplot(data = dfmainfp, x="Lesion Size", y="Lesion Sizes [mL]", hue="Model", dodge=True, alpha=.25, zorder=1, ax=ax[0], palette = [color_list[0], color_dark[1], color_list[2], color_list[3]])
# Show the conditional means, aligning each pointplot in the
# center of the strips by adjusting the width allotted to each
# category (.8 by default) by the number of hue levels
sns.pointplot(data = dfmainfp, x="Lesion Size", y="Lesion Sizes [mL]", hue="Model", dodge=.532, join=False, palette = [color_dark[0], color_dark[1], color_dark[2], color_dark[3]] , markers="d", scale=.75, errorbar=None, ax=ax[0])


# Improve the legend put legend outside
handles, labels = ax[0].get_legend_handles_labels()
ax[0].legend(handles=handles[4:], labels=labels[4:], loc='lower center', bbox_to_anchor=(0.5, 1.05))
# Finalize the figure
ax[0].set_xscale('log')
ax[0].set_xlabel('Lesion Size [log]')


sns.despine(bottom=True, left=True)
# Show each observation with a scatterplot
sns.stripplot(data = dfmain, x="Recall", y="Lesion Sizes [mL]", hue="Model", dodge=True, alpha=.25, zorder=1, ax=ax[1], palette = [color_list[0], color_dark[1], color_list[2], color_list[3]])
# Show the conditional means, aligning each pointplot in the
# center of the strips by adjusting the width allotted to each
# category (.8 by default) by the number of hue levels
sns.pointplot(data = dfmain, x="Recall", y="Lesion Sizes [mL]", hue="Model", dodge=.532, join=False, palette = [color_dark[0], color_dark[1], color_dark[2], color_dark[3]], markers="d", scale=.75, errorbar=None, ax=ax[1])
# Improve the legend put legend outside
# handles, labels = ax[1].get_legend_handles_labels()
# ax[1].legend(handles=handles[2:], labels=labels[2:], loc='lower center', bbox_to_anchor=(0.5, 1.05))
# b
# Finalize the figure

sns.despine(bottom=True, left=True)
# Show each observation with a scatterplot
sns.stripplot(data = dfmainfp, x="Precision", y="Lesion Sizes [mL]", hue="Model", dodge=True, alpha=.25, zorder=1, ax=ax[2], palette = [color_list[0], color_dark[1], color_list[2], color_list[3]])
# Show the conditional means, aligning each pointplot in the
# center of the strips by adjusting the width allotted to each
# category (.8 by default) by the number of hue levels
sns.pointplot(data = dfmainfp, x="Precision", y="Lesion Sizes [mL]", hue="Model", dodge=.532, join=False, palette = [color_dark[0], color_dark[1], color_dark[2],  color_dark[3]], markers="d", scale=.75, errorbar=None, ax=ax[2])
# Improve the legend put legend outside
# handles, labels = ax[1].get_legend_handles_labels()
# ax[2].legend(handles=handles[2:], labels=labels[2:], loc='lower center', bbox_to_anchor=(0.5, 1.05))
# Finalize the figure

# remove legend for ax 1 and 2
ax[1].legend_.remove()
ax[2].legend_.remove()
# remove y label for ax 0 and 2
plt.tight_layout()


plt.savefig('./plots_and_metrics_wmh/lesion_plots/lesion_recall_precision_test_fixed_threshold.jpg', dpi=300)
plt.close()


# remove ID
# dfmain = dfmain.drop(columns=['ID'])
# sns.pairplot(dfmain, hue="Model", corner=True, diag_kind="hist", plot_kws=dict(s=80, edgecolor="white", linewidth=2.5))
# plt.tight_layout()
# plt.savefig('./plots_and_metrics_wmh/lesion_plots/lesion_recall.png', dpi=300)

# fig, ax = plt.subplots(1, 2, figsize=(10, 6))
# sns.despine(fig)
# sns.histplot(data=dfmainfp, x="Lesion Sizes [mL]", hue = "Model", kde=False, multiple="stack", palette="light:m_r", edgecolor=".3", linewidth=.5, ax=ax[0])
# sns.histplot(data=dfmain[dfmain["Model"] == "2.5D U-Net"], x="Lesion Sizes [mL]", kde=False, ax=ax[1]) # Get ground truth lesions
# ax[0].set_title("Lesions detected")
# ax[1].set_title("Ground truth lesions")
# # tilt x-axis labels
# for tick in ax[0].get_xticklabels():
#     tick.set_rotation(45)
# for tick in ax[1].get_xticklabels():
#     tick.set_rotation(45)
# plt.tight_layout()
# plt.savefig('./plots_and_metrics_wmh/lesion_plots/lesion_recall.png', dpi=300)
# plt.close()



# # scatterplot with density overlap for close points

# # sns.set(style="white", color_codes=True)
# sns.histplot(data=df, x="Size Bracket", hue="Model", multiple="stack", bins=10, kde=False)
# plt.savefig('lesion_recall.png', dpi=300)


# sns.histplot(data=df, x="Lesion Size", hue="Model", multiple="stack", bins=20, kde=True)
# plt.savefig("lesion_recall.png")
# sns.kdeplot(data=df25, x="Lesion Size", y="Recall", levels=15, color="red", linewidths=1, fill = False, alpha = 0.1)
# sns.kdeplot(data=dfbay, x="Lesion Size", y="Recall", levels=15, color="blue", linewidths=1, fill = False, alpha = 0.1)
# sns.kdeplot(data=dfnn, x="Lesion Size", y="Recall", levels=15, color="green", linewidths=1, fill = False, alpha = 0.1)
# # same colors
# sns.scatterplot(data=df, x="Lesion Size", y="Recall", hue="Model", alpha=1, s=10, palette=["red", "blue", "green"])
# # limit axis
# plt.ylim(0, 1)
# plt.savefig("lesion_recall.png")
