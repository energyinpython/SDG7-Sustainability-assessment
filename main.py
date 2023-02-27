# ICECCME 2022 - Maldives
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

from pyrepo_mcda.mcda_methods import TOPSIS, ARAS, EDAS, CODAS, PROMETHEE_II
from pyrepo_mcda import distance_metrics as dists
from pyrepo_mcda import weighting_methods as mcda_weights
from pyrepo_mcda import normalizations as norms
from pyrepo_mcda import correlations as corrs
from pyrepo_mcda.additions import rank_preferences

# from pymcdm import weights as mcdm_weights
# from pymcdm import methods as mcdm_methods



# bar chart
def plot_barplot(df_plot, x_name, y_name, title, sth = ''):
    """
    Display stacked column chart of weights for criteria for `x_name == Weighting methods`
    and column chart of ranks for alternatives `x_name == Alternatives`

    Parameters
    ----------
        df_plot : dataframe
            dataframe with criteria weights calculated different weighting methods
            or with alternaives rankings for different weighting methods
        x_name : str
            name of x axis, Alternatives or Weighting methods
        y_name : str
            name of y axis, Ranks or Weight values
        title : str
            name of chart title, Weighting methods or Criteria

    Examples
    ----------
    >>> plot_barplot(df_plot, x_name, y_name, title)
    """
    
    list_rank = np.arange(1, len(df_plot) + 2, 2)
    stacked = False
    width = 0.8
    ncol = 3
    if sth == 'prom':
        ncol = 6
    
    
    # blueviolet choose colors
    if sth == 'mcda':
        colors = ['#d62728', 'greenyellow', '#1f77b4']
        ax = df_plot.plot(kind='bar', width = width, stacked=stacked, color = colors, edgecolor = 'black', figsize = (10,4))
    else:
        ax = df_plot.plot(kind='bar', width = width, stacked=stacked, edgecolor = 'black', figsize = (10,4))
    ax.set_xlabel(x_name, fontsize = 12)
    ax.set_ylabel(y_name, fontsize = 12)

    ax.set_yticks(list_rank)
    ax.set_xticklabels(df_plot.index, rotation = 'horizontal')
    ax.tick_params(axis = 'both', labelsize = 12)

    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
    ncol=ncol, mode="expand", borderaxespad=0., edgecolor = 'black', title = title, fontsize = 11)

    ax.grid(True, linestyle = '--')
    ax.set_axisbelow(True)
    plt.tight_layout()
    plt.savefig('./results/bar_chart' + '_' + sth + '.png')
    plt.show()


# heat maps with correlations
def draw_heatmap(df_new_heatmap, title, sth = ''):
    """
    Visualization method to display heatmap with correlations of compared rankings generated using different methods
    
    Parameters
    ----------
        data : DataFrame
            DataFrame with correlation values between compared rankings
        title : str
            title of chart containing name of used correlation coefficient
    
    Examples
    ---------
    >>> draw_heatmap(df_new_heatmap, title)
    """
    if sth == 'promethee':
        plt.figure(figsize = (7, 7))
    elif sth == 'validation':
        plt.figure(figsize = (6, 5))
    else:
        plt.figure(figsize = (5, 4))
    sns.set(font_scale = 1.4)
    heatmap = sns.heatmap(df_new_heatmap, annot=True, fmt=".4f", cmap="RdYlGn",
                          linewidth=0.5, linecolor='w')
    plt.yticks(va="center")
    if sth == 'promethee':
        plt.xlabel('PROMETHEE II preference functions')
    else:
        plt.xlabel('MCDA methods')
    if sth == 'validation':
        plt.ylabel('PROMETHEE II preference functions')
    plt.title('Correlation: ' + title)
    plt.tight_layout()
    title = title.replace("$", "")
    plt.savefig('./results/' + 'correlations_' + title + '_' + sth + '.png')
    plt.show()


# Create dictionary class
class Create_dictionary(dict):
  
    # __init__ function
    def __init__(self):
        self = dict()
          
    # Function to add key:value
    def add(self, key, value):
        self[key] = value


def main():
    folder_name = './dataset'
    filename = 'data_2020.csv'
    pathfile = os.path.join(folder_name, filename)
    data = pd.read_csv(pathfile, index_col='Country')
    df = data.iloc[:len(data) - 1, :]

    # types
    types = data.iloc[len(data) - 1, :].to_numpy()
    
    path_symbols = os.path.join(folder_name, 'country_symbols.csv')
    df_symbols = pd.read_csv(path_symbols, index_col='Country')
    symbols = list(df_symbols['Symbol'])

    # matrix
    matrix = df.to_numpy()

    # CRITIC weighting
    # bo pozadana normalizacja min-max a my mamy negative values w decision matrix
    # weights
    weights = mcda_weights.critic_weighting(matrix)

    # print(weights)
    # print(np.sum(weights))

    preferences = pd.DataFrame(index = symbols)
    rankings = pd.DataFrame(index = symbols)

    # TOPSIS
    topsis = TOPSIS(normalization_method=norms.minmax_normalization, distance_metric=dists.euclidean)
    pref = topsis(matrix, weights, types)
    rank = rank_preferences(pref, reverse = True)
    preferences['TOPSIS'] = pref
    rankings['TOPSIS'] = rank

    # ARAS
    aras = ARAS(normalization_method=norms.minmax_normalization)
    pref = aras(matrix, weights, types)
    rank = rank_preferences(pref, reverse=True)
    preferences['ARAS'] = pref
    rankings['ARAS'] = rank

    # # EDAS
    # edas = EDAS()
    # pref = edas(matrix, weights, types)
    # rank = rank_preferences(pref, reverse=True)
    # preferences['EDAS'] = pref
    # rankings['EDAS'] = rank

    # CODAS
    codas = CODAS(normalization_method=norms.minmax_normalization, distance_metric=dists.euclidean)
    pref = codas(matrix, weights, types)
    rank = rank_preferences(pref, reverse=True)
    preferences['CODAS'] = pref
    rankings['CODAS'] = rank

    # save rankings and preferences in csv
    rankings.to_csv('./results/rankings.csv')
    preferences.to_csv('./results/preferences.csv')

    # bar chart MCDA
    plot_barplot(rankings, 'Countries', 'Rank', 'MCDA methods', 'mcda')

    # korelacje
    method_types = list(rankings.columns)
    dict_new_heatmap_rw = Create_dictionary()
    for el in method_types:
        dict_new_heatmap_rw.add(el, [])

    # heatmaps for correlations coefficients
    for i, j in [(i, j) for i in method_types[::-1] for j in method_types]:
        dict_new_heatmap_rw[j].append(corrs.weighted_spearman(rankings[i], rankings[j]))

    df_new_heatmap_rw = pd.DataFrame(dict_new_heatmap_rw, index = method_types[::-1])
    df_new_heatmap_rw.columns = method_types

    # correlation matrix with rw coefficient
    draw_heatmap(df_new_heatmap_rw, r'$r_w$')

    # ==============================================================================
    # walidacja z PROMETHEE II

    rankings_prom = pd.DataFrame(index = symbols)

    promethee_II = PROMETHEE_II()
    list_pf_functions = [
        promethee_II._usual_function,
        promethee_II._ushape_function,
        promethee_II._vshape_function,
        promethee_II._level_function,
        promethee_II._linear_function,
        promethee_II._gaussian_function
    ]


    for pf in list_pf_functions:
        preference_functions = [pf for el in range(len(weights))]

        pref = promethee_II(matrix, weights, types, preference_functions = preference_functions)
        rank = rank_preferences(pref, reverse=True)
        pf_name = pf.__name__
        pf_name = pf_name.replace("_function", "")
        pf_name = pf_name.replace("_", "")
        pf_name = pf_name.replace("shape", "-shape")
        pf_name = pf_name.capitalize()
        rankings_prom[pf_name] = rank

    # save rankings in csv
    rankings_prom.to_csv('./results/rankings_prom.csv')

    # korelacje
    method_types = list(rankings_prom.columns)
    dict_new_heatmap_rw = Create_dictionary()
    for el in method_types:
        dict_new_heatmap_rw.add(el, [])

    # heatmaps for correlations coefficients
    for i, j in [(i, j) for i in method_types[::-1] for j in method_types]:
        dict_new_heatmap_rw[j].append(corrs.weighted_spearman(rankings_prom[i], rankings_prom[j]))

    df_new_heatmap_rw = pd.DataFrame(dict_new_heatmap_rw, index = method_types[::-1])
    df_new_heatmap_rw.columns = method_types

    # correlation matrix with rw coefficient
    draw_heatmap(df_new_heatmap_rw, r'$r_w$', 'promethee')

    # bar chart promethee II
    matplotlib.rc_file_defaults()
    plot_barplot(rankings_prom, 'Countries', 'Rank', 'PROMETHEE II preference functions', 'promethee')

    df_valid = pd.DataFrame(0, index = rankings_prom.columns, columns = rankings.columns)

    for i, col_prom in enumerate(list(rankings_prom.columns)):
        for j, col in enumerate(list(rankings.columns)):
            df_valid.iloc[i, j] = corrs.weighted_spearman(rankings_prom[col_prom], rankings[col])

    # correlation matrix with rw coefficient
    draw_heatmap(df_valid, r'$r_w$', 'validation')


    # # prom II test
    # print('-----')
    # print('PROM')
    # print(rankings_prom)


    # rankings_prom = pd.DataFrame(index = symbols)
    # u = np.sqrt(np.sum(np.square(np.mean(matrix, axis = 0) - matrix), axis = 0) / matrix.shape[0])
    # p = 2 * u
    # q = 0.5 * u

    # preference_functions = ['usual', 'ushape', 'vshape', 'level', 'vshape_2']
    # for pf in preference_functions:

    #     function = mcdm_methods.PROMETHEE_II(preference_function = pf)
    #     result = function(matrix, weights, types, p=p, q=q)
    #     rankings_prom[pf] = rank_preferences(result, reverse = True)

    # print('-----')
    # print(rankings_prom)

    

if __name__ == '__main__':
    main()
