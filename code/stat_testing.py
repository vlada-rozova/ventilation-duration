import os
import pandas as pd
import numpy as np
from scipy import stats 

from matplotlib import pyplot as plt
import seaborn as sns

        
def ttest(df, by, col, equal_var, transform=None, verbose=True):
    """
    Run a t-test or a Welch t-test for unequal variances.
    """
    x_start = []
    x_end = []
    signif = []
    
    #group = df[by].unique().sort_values()
    group = df[by].unique()
    for i in range(len(group) - 1):
        x1 = df.loc[df[by] == group[i], col]
        x2 = df.loc[df[by] == group[i + 1], col]
        
        if transform == 'log':
            x1 = np.log(x1)
            x2 = np.log(x2)
        elif transform == 'boxcox':
            x1,_ = stats.boxcox(x1)
            x2,_ = stats.boxcox(x2)
            
        _, p = stats.ttest_ind(x1, x2, equal_var=equal_var, nan_policy='omit')
        if p < 0.05:
            x_start.append(i)
            x_end.append(i + 1)
            if p < 0.001:
                sign = '***'
            elif p < 0.01:
                sign = '**'
            elif p < 0.05:
                sign = '*'
            signif.append(sign)
        else:
            sign = ''
            
        if verbose:
            if equal_var:
                print("A two-tailed t-test on samples {} vs {}. {} \t p-value = {:.2}."
                      .format(group[i], group[i + 1],  sign, p))
            else:
                print("Welch's unequal variances t-test on samples {} vs {}. {} \t p-value = {:.2}."
                      .format(group[i], group[i + 1],  sign, p))
            
    return x_start, x_end, signif


def mannwhitneytest(df, by, col, alternative, transform=None):
    """
    Run a Mann-whitney U test
    """
    x_start = []
    x_end = []
    signif = []
    
    group = df[by].sort_values().unique()
    for i in range(len(group) - 1):
        x1 = df.loc[df[by] == group[i], col]
        x2 = df.loc[df[by] == group[i + 1], col]
        
        if transform == 'log':
            x1 = np.log(x1)
            x2 = np.log(x2)
        elif transform == 'boxcox':
            x1,_ = stats.boxcox(x1)
            x2,_ = stats.boxcox(x2)
            
        _, p = stats.mannwhitneyu(x1, x2, alternative=alternative)
        if p < 0.05:
            x_start.append(i)
            x_end.append(i + 1)
            if p < 0.001:
                sign = '***'
            elif p < 0.01:
                sign = '**'
            elif p < 0.05:
                sign = '*'
            signif.append(sign)
        else:
            sign = ''

        print("A {} Mann Whitney test on samples {} vs {}. {} \t p-value = {:.2}."
              .format(alternative, group[i], group[i + 1],  sign, p))
        
    return x_start, x_end, signif
        
def stat_annot(df, by, col, x_start, x_end, signif, ylim, kind='barplot'):
    """
    Add annotation to show the results of statistical testing.
    """
    s = df[by].sort_values().unique()
    h = (ylim[1] - ylim[0])/50
    
    for x1, x2, label in zip(x_start, x_end, signif):

        if kind == 'barplot':
            y = max(df.loc[df[by] == s[x1], col].mean() + df.loc[df[by] == s[x1], col].std(), 
                    df.loc[df[by] == s[x2], col].mean() + df.loc[df[by] == s[x2], col].std()) + h
        elif kind == 'boxplot':
            y = max(upper_whisker(df.loc[df[by] == s[x1], col]), 
                    upper_whisker(df.loc[df[by] == s[x2], col])) + h

        plt.plot([x1+0.05, x1+0.05, x2-0.05, x2-0.05], [y, y + h, y + h, y], 
                 lw=1.5, color='k');

        plt.text((x1 + x2) * 0.5, y + h, s=label, 
                 ha='center', va='bottom', 
                 color='k', fontsize=20);
        plt.ylim(ylim)
        
        
def upper_whisker(x):
    iqr = x.quantile(0.75) - x.quantile(0.25)
    return x.quantile(0.75) + 1.5 * iqr