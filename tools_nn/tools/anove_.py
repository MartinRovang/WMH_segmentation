from scipy import stats
import pandas as pd
import numpy as np


group1 = pd.read_csv('/mnt/CRAI-NAS/all/martinsr/NNunet/report/brno/stats_2022-10-23_brno_bayesian.csv')
group2 = pd.read_csv('/mnt/CRAI-NAS/all/martinsr/NNunet/report/brno/stats_2022-10-25_brno_25DUNet.csv')
group3 = pd.read_csv('/mnt/CRAI-NAS/all/martinsr/NNunet/report/brno_fixed_threshold/stats_2022-10-30_brno_nnU-Net.csv')


# get dsc values
group1 = group1['dsc'].values
group2 = group2['dsc'].values
group3 = group3['dsc'].values
# if there are any zeros, replace them with a small value
# import pingouin as pg

# stack data by columns from group1


# aov = pg.anova(dv='dsc', between=['model', 'participant'], data=df, detailed=True)
# print(aov)

def anova_friedman(data, alpha=0.05):
    """Perform Related-Samples Friedman's Two-Way Analysis of Variance by Ranks"""
    # calculate Friedman statistic
    stat, p = stats.friedmanchisquare(*data)
    # interpret test-statistic
    print('Friedman statistic=%.3f, p=%.3f' % (stat, p))
    # interpret p-value
    if p > alpha:
        print('Same distributions (fail to reject H0)')
    else:
        print('Different distributions (reject H0)')
    
    # pairwise comparison
    groups = ["group1", "group2", "group3"]
    for i in range(len(data)):
        for j in range(i+1, len(data)):
            stat, p = stats.wilcoxon(data[i], data[j])
            # bonferroni corrected p-value
            p = p * 3
            print(f'{groups[i]}-{groups[j]} Wilcoxon statistic=%.3f, p=%.3f' % (stat, p))
            if p > alpha:
                print('Same distributions (fail to reject H0)')
            else:
                print('Different distributions (reject H0)')


anova_friedman([group1, group2, group3])