import math

import networkx
import numpy as np
import scanpy as sc
import pandas as pd
from os.path import exists
import scanpy.external as sce
import seaborn as sns
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.pyplot import legend
from statsmodels.formula.api import ols
from statsmodels.stats.multitest import fdrcorrection
import statsmodels.api as sm
from scipy import stats
from scipy.stats import mannwhitneyu
from scipy.stats import hypergeom
import plotly.express as px
# See https://stackoverflow.com/questions/52771328/plotly-chart-not-showing-in-jupyter-notebook
# Import the necessaries libraries
import plotly.offline as pyo
import plotly.graph_objs as go

# Set notebook mode to work in offline
pyo.init_notebook_mode()
# Local scripts
# Follow the instruction at https://stackoverflow.com/questions/34478398/import-local-function-from-a-module-housed-in-another-directory-with-relative-im
# So that it can work for both local script and notebook
import os
import sys
from typing import Union
# For doing some pathway-based analysis
from scpy4reactome import pathway_analyzer as pa
from pyvis.network import Network
import pickle
import xml.etree.ElementTree as et

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
from scripts.Python.PlotGenerators import run_umap_via_paga

random_state = 17051256


def save_list(list, file_name):
    file = open(file_name, 'wb')
    pickle.dump(list, file)
    file.close()


def load_list(file_name):
    file = open(file_name, 'rb')
    list = pickle.load(file)
    file.close()
    return list


def load_data(h5ad_file: str,
              data_file: str,
              meta_file: str) -> sc.AnnData:
    """
    Load the data file and meta file into an AnnData object.
    :param data_file:
    :param meta_file:
    :return:
    """
    # Check if h5ad file existing
    if exists(h5ad_file):
        anndata = sc.read_h5ad(h5ad_file)
        return anndata
    # load the data file into an AnnData object
    andata = sc.read_csv(data_file)
    andata = andata.transpose()  # Make sure obs are cells and vars are genes
    # Load the meta file: the first column will be used as the index as in R
    meta = pd.read_csv(meta_file)
    andata.obs = andata.obs.join(meta)
    andata.write_h5ad(h5ad_file)
    return andata


def pre_process(adata: sc.AnnData):
    """
    Pre-processs adata as usual.
    :param adata:
    :return:
    """
    # Most likely these filterings have been performed in the filtered matrix
    # Anyhow, just do them again.
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
    adata.var['mt'] = adata.var_names.str.startswith('MT-') | adata.var_names.str.startswith('mt-')
    sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
    # Have not done any filtering based on total gene numbers
    adata = adata[adata.obs.pct_counts_mt < 20, :]
    sc.pp.normalize_total(adata, 1E+4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata)
    return adata


def run_harmony(adata, batch_key='replicate.ID'):
    # Make sure obsm has batch_key value
    adata.obsm[batch_key] = adata.obs[batch_key]
    sce.pp.harmony_integrate(adata, batch_key)
    sc.pp.neighbors(adata, use_rep='X_pca_harmony', random_state=random_state)
    run_umap_via_paga(adata, use_rep='X_pca_harmony')
    return adata


def plot_paga(anndata: sc.AnnData,
              threshold: float = None,
              color: list = ['leiden']):
    sc.pl.paga(anndata,
               threshold=threshold,
               color=color,
               pos=anndata.uns['paga']['pos'])  # Don't make any difference by assigning root


def plot_all_samples(adata: sc.AnnData):
    """
    Plot all samples
    :param adata:
    :return:
    """
    samples = _get_samples(adata)
    for sample in samples:
        print(sample)
        plot_sample(adata, sample)


def _get_samples(adata: sc.AnnData):
    samples = adata.obs['replicate.ID'].unique().to_list()
    # Sort to place a pair of samples together
    samples.sort(key=lambda x: (x.split('_')[0] + '_' + x.split('_')[2] + '_' + x.split('_')[1]))
    return samples;


def plot_sample(adata: sc.AnnData,
                sample: str):
    """
    Plot a sample in UMAP
    :param adata:
    :param sample: should be one of replicate.ID
    :return:
    """
    if sample not in adata.obs['replicate.ID'].unique():
        raise ValueError('{} is not a valid replicate.ID.'.format(sample))
    selected_samples = adata.obs['replicate.ID'] == sample
    adata_sample = adata[selected_samples,]
    sc.pl.umap(adata_sample, color=['leiden', 'cell.ident'], legend_loc='on data')


def analyze_cell_distributions(adata: sc.AnnData,
                               cell_field: str = 'cell.ident'):
    """
    This method is used to analyze the cell type distributions for individual samples. The distributions
    are converted into percentage, plotted, and checked via ANOVA.
    :param adata:
    :return:
    """
    cell_types = adata.obs[cell_field].unique().to_list()
    # Sorted it for each view
    cell_types.sort()
    # Get samples
    samples = _get_samples(adata)
    # Have a DataFrame to hold results
    cell_types_df = pd.DataFrame(columns=cell_types, index=samples)
    for sample in samples:
        # Filtering to this sample
        adata_sample = adata[adata.obs['replicate.ID'] == sample,]
        total_cells = adata_sample.shape[0]
        cell_types_sample = adata_sample.obs[cell_field]
        for cell_type in cell_types:
            total_cell_type = sum(cell_types_sample == cell_type)
            cell_types_df.loc[sample, cell_type] = total_cell_type / total_cells
    plot_cell_type_distribution(cell_types_df)
    return cell_types_df


def plot_cell_type_distribution(cell_type_df: pd.DataFrame):
    """
    Plot the cell type distribution for each sample and each cell type.
    :param cell_type_df:
    :return:
    """
    # Plot these cell types
    cell_types = cell_type_df.columns.to_list()
    # Add another column called type
    cell_type_df['Sample_Category'] = cell_type_df.index
    cell_type_df['Sample_Category'] = cell_type_df['Sample_Category'].map(
        lambda x: x.split('_')[0] + '_' + x.split('_')[1])
    # Get the repeat for easy label
    cell_type_df['Replication'] = cell_type_df.index
    cell_type_df['Replication'] = cell_type_df['Replication'].map(lambda x: x.split('_')[2])
    for cell_type in cell_types:
        # sns.catplot(x='Sample_Category', y=cell_type, hue='Replication', data=cell_type_df, kind='bar')
        sns.boxplot(x='Sample_Category', y=cell_type, data=cell_type_df, color='#99C2C2')
        sns.swarmplot(x='Sample_Category', y=cell_type, data=cell_type_df, color='#7D0013')
        plt.show()  # Have to call this to force plots to display
        anova_results = _perform_annova(cell_type_df, cell_type)
        print(anova_results)
    plot_stacked_bar(cell_type_df, cell_types)


def plot_stacked_bar(cell_type_df: pd.DataFrame,
                     cell_types: list):
    # melt it
    cell_type_df.set_index(keys='Sample_Category', drop=True, inplace=True)
    cell_type_df_sum = cell_type_df.groupby(by=cell_type_df.index).sum()
    cell_type_df_sum.plot.bar(stacked=True, legend='reverse')
    # Control the legend location
    # TODO: The above legend's reverse cannot work here.
    plt.legend(loc='center right', bbox_to_anchor=(1.25, 0.5), fontsize=7)


def _perform_annova(cell_type_df: pd.DataFrame,
                    cell_type_col: str):
    # Make sure the colum names that are not compatible to variable names can be used here
    # See https://patsy.readthedocs.io/en/latest/builtins-reference.html#patsy.builtins.Q
    formula = 'Q("{}") ~ C(Sample_Category)'.format(cell_type_col)
    # Make sure this is numeric. Otherwise, it will not work
    cell_type_df[cell_type_col] = pd.to_numeric(cell_type_df[cell_type_col])
    model = ols(formula, data=cell_type_df).fit()
    aov_table = sm.stats.anova_lm(model, typ=2)
    return aov_table


def cross_sample_analysis(adata: sc.AnnData,
                          samples1: Union[str, list],
                          samples2: Union[str, list],
                          sample_key: str = 'geno_treat',
                          obs_key: str = 'leiden',
                          score_type: str = None,
                          need_diff: bool = False) -> pd.DataFrame:
    """
    Perform a differential analysis between two samples for gene expression, pathway score or
    TF score. The current implementation uses Mann-Whitney U Test (https://machinelearningmastery.com
    /nonparametric-statistical-significance-tests-in-python/)
    :param adata:
    :param sample1:
    :param sample2:
    :param sample_key: either 'geno-treat' or 'replicate.ID'
    :param obs_key: either cell.ident or leiden
    :param score_type: gene expression if None
    :param need_diff: calculate median difference if true between two samples
    :return:
    """
    if obs_key not in ['leiden', 'cell.ident']:
        raise ValueError('obs_key must be either leiden or cell.ident')
    if (score_type is not None) and (score_type not in adata.obsm.keys()):
        raise ValueError('{} is not in obs.keys.'.format(score_type))
    if isinstance(samples1, str):
        samples1 = [samples1]
    if isinstance(samples2, str):
        samples2 = [samples2]
    adata_samples_1 = adata[adata.obs[sample_key].isin(samples1),]
    adata_samples_2 = adata[adata.obs[sample_key].isin(samples2),]
    # DataFrame to hold the results
    row_names = adata.obs[obs_key].unique().sort_values()
    cols = [obs_key, 'feature', 'stat', 'p-value']
    if need_diff:
        cols.append('median_diff')
    result_df = pd.DataFrame(columns=cols)
    features = adata.obsm[score_type] if (score_type is not None) else adata.var_names
    row = 0
    for row_name in row_names:
        which_cells = adata_samples_1.obs[obs_key] == row_name
        adata_samples_1_row = adata_samples_1[which_cells,]
        which_cells = adata_samples_2.obs[obs_key] == row_name
        adata_samples_2_row = adata_samples_2[which_cells,]
        for feature in features:
            scores1 = adata_samples_1_row.obsm[score_type][feature] if (score_type is not None) else \
                adata_samples_1_row.obs[feature]
            scores2 = adata_samples_2_row.obsm[score_type][feature] if (score_type is not None) else \
                adata_samples_2_row.obs[feature]
            try:
                test_result = mannwhitneyu(scores1, scores2)
                row_values = [row_name, feature, test_result[0], test_result[1]]
                if need_diff:
                    row_values.append(np.median(scores1) - np.median(scores2))
                result_df.loc[row] = row_values
                row += 1
            except ValueError as e:
                print("{} and {}: {}.".format(row_name, feature, e))
    # Add a fdr correction
    # Should get the second returned value
    result_df['fdr'] = fdrcorrection(result_df['p-value'])[1]
    # Sort the final result based on fdr
    result_df.sort_values(by='fdr', inplace=True)
    return result_df


def violin_plots(adata: sc.AnnData,
                 tfs: list,
                 obs_key: str = 'geno_treat',
                 sample_types: list = ['WT_unt', 'KO_unt']):
    for tf in tfs:
        if tf in adata.obs_keys():
            adata.obs = adata.obs.drop(columns=tf)
        fig, axs = plt.subplots(ncols=2)
        fig.set_figwidth(16)  # This seems a good width
        violin_plot(adata, adata.obs['leiden'],
                    adata.obsm[pa.TF_AUCELL_KEY][tf],
                    obs_key,
                    sample_types,
                    ax=axs[0])
        violin_plot(adata, adata.obs['leiden'],
                    adata.obs_vector(tf),
                    obs_key,
                    sample_types,
                    ax=axs[1])


def violin_plot(adata: sc.AnnData,
                x,
                y,
                obs_key: str,
                hue: list,
                ax=None):
    which_samples = adata.obs[obs_key].isin(hue)
    figure = sns.violinplot(x=x[which_samples],
                            y=y[which_samples],
                            hue=adata.obs[obs_key][which_samples].to_list(),
                            split=True,
                            ax=ax)
    return figure
    # plt.show(figure)


def plot_pathways(pathway_df: pd.DataFrame,
                  feature_col: str = None):
    """
    Put pathways in a scatter plot based on top level pathways.
    The code here is based on
    :param pathway_df: the data frame
    :param feature_col: none for using -Log10(FDR).
    :return:
    """
    pathway_hr_file = '../resources/Pathway_List_In_Hirerarchy_Release_79.txt'
    pathway_hr_df = pd.read_csv(pathway_hr_file, sep='\t')
    # Exclude disease pathways
    disease_pathways = pathway_hr_df['Top_Level_Pathway'] == 'Disease'
    pathway_hr_df = pathway_hr_df[~disease_pathways]
    # Apparently the order is not right there
    pathway_hr_df = pathway_hr_df.iloc[::-1]
    pathway_hr_df = pathway_hr_df.merge(pathway_df, how='left', left_on='Pathway', right_on='feature')
    if feature_col is None:
        feature_col = _add_log10_fdr_col(pathway_hr_df)
    fig = px.scatter(pathway_hr_df,
                     x='Pathway',
                     y=feature_col,
                     color='Top_Level_Pathway')
    fig.update_xaxes(showticklabels=False)
    fig.show()
    return pathway_hr_df


def plot_tfs(tf_df: pd.DataFrame,
             feature_col: str = None):
    if feature_col is None:
        _add_log10_fdr_col(tf_df)
        feature_col = '-Log10(FDR)'
    fig = px.scatter(tf_df,
                     x='feature',
                     labels={'feature': 'TF'},
                     y=feature_col)
    fig.show()
    return plot_tfs


def _add_log10_fdr_col(df: pd.DataFrame):
    feature_col = '-Log10(FDR)'
    # Need to avoid log(0)
    fdr = df['fdr']
    fdr_min = min(fdr[fdr > 0])
    # print(fdr_min)
    df[feature_col] = df['fdr'].map(
        lambda fdr: -math.log10(fdr) if fdr > fdr_min else (np.NAN if np.isnan(fdr) else -math.log10(fdr_min / 10)))
    return feature_col


def analyze_correlation_via_glm(adata: sc.AnnData,
                                pathway: str,
                                tfs: list):
    """
    Perform a correlation analysis from tfs to a pathway
    :param adata:
    :param pathway:
    :param tfs:
    :return:
    """
    pathway_score = adata.obsm[pa.AUCELL_KEY][pathway]
    tfs_scores = adata.obsm[pa.TF_AUCELL_KEY][tfs]
    x_scores = sm.add_constant(tfs_scores, prepend=False)
    glm_model = sm.GLM(pathway_score, x_scores)
    glm_result = glm_model.fit()
    return glm_result


def analyze_correlations_via_glm(adata: sc.AnnData,
                                 pathways: list,
                                 tfs: list) -> pd.DataFrame:
    """
    Perform GLM correlation analysis for a set of pathways
    :param adata:
    :param pathways:
    :param tfs:
    :return:
    """
    result_df = pd.DataFrame()
    for pathway in pathways:
        glm_result = analyze_correlation_via_glm(adata, pathway, tfs)
        result_df[pathway + '_param'] = glm_result.params
        result_df[pathway + '_pvalue'] = glm_result.pvalues
    return result_df


def build_tfs_network(tfs: list,
                      tf2targets: dict,
                      check_with_tf_cor: bool = True,
                      tf_cor_cutoff: float = 0.25,
                      adata: sc.AnnData = None,
                      network: nx.DiGraph = None) -> nx.DiGraph:
    """
    Build TF/target network for a list of tfs.
    :param tfs:
    :param tf2targets:
    :return:
    """
    if network is None:
        network = nx.DiGraph()
    # To scale the value so that we have the same value for tf -> pathway
    max_value = None
    all_values = [math.fabs(v) for k, v in nx.get_edge_attributes(network, 'value').items()]
    if len(all_values) > 0:
        max_value = max(all_values)
    for tf1 in tfs:
        for tf2 in tfs:
            # We may see feedback loops to tf itself or others. Therefore both check
            if tf2 in tf2targets[tf1]:
                if check_with_tf_cor:
                    if adata is not None and pa.TF_AUCELL_KEY in adata.obsm.keys():
                        # Calculate correlation
                        tf1_aucell = adata.obsm[pa.TF_AUCELL_KEY][tf1]
                        tf2_aucell = adata.obsm[pa.TF_AUCELL_KEY][tf2]
                        # Ignore p-value for the time being here
                        cor_value = stats.spearmanr(tf1_aucell, tf2_aucell)[0]
                        if max_value is not None:
                            cor_value *= max_value # The maximum should be the same
                        if cor_value > tf_cor_cutoff:
                            network.add_edge(tf1, tf2, color='#8FBC8F',
                                             annotation='tf_tf_activation', value=cor_value)  # dark gree for positiveTF edges
                        elif cor_value < -tf_cor_cutoff:
                            network.add_edge(tf1, tf2, color='#A52A2A',
                                             annotation='tf_tf_inhibition', value=-cor_value)  # brown for negative TF edges
                else:
                    network.add_edge(tf1, tf2, annotation='tf_tf_interaction', color='#A52A2A') # brown for TF edges
    return network


def add_pathway_to_tf_network(pathways: list,
                              tfs: list,
                              pathway2genes: dict,
                              network: nx.DiGraph) -> nx.DiGraph:
    """
    Build a network or add new edges bewteen pathways and tfs. If a pathway has a TF annotated, an edge
    will be added between this pathway to TF (from pathway to TF). This is a highly simplified version.
    Manual check should be performed in the future to ensure these links are correct.
    :param pathways:
    :param tfs:
    :param pathway2genes:
    :param netowrk:
    :return:
    """
    # This is a little bit more complicated. We want to push this link as low as possible.
    # Start with TFs
    for tf in tfs:
        selected_pathways = []
        for pathway in pathways:
            pathway_genes = pathway2genes[pathway]
            if tf in pathway_genes:
                selected_pathways.append(pathway)
        # To a prunning to remove other pathways that can be covered by child pathways
        if len(selected_pathways) > 0:
            for pathway in selected_pathways:
                if not _is_in_child_pathways(pathway, selected_pathways, network):
                    network.add_edge(pathway, tf, annotation='pathway_tf_annotation', color='#E9967A') #Point from pathway to tf. Dark Salmon
    return network


def _is_in_child_pathways(pathway, selected_pathways, network) -> bool:
    in_edges = network.in_edges(pathway)
    for u, v in in_edges:
        if network.nodes[u]['type'] == 'TF':
            continue
        if u in selected_pathways:
            return True
        is_included = _is_in_child_pathways(u, selected_pathways, network)
        if is_included:
            return True
    return False


def build_network_for_tfs_pathways(pathways_tf_cors: pd.DataFrame,
                                   param_cutoff: float = 0.1,
                                   p_value_cutff: float = 1.0E-6) -> nx.DiGraph:
    """
    Build  network from TFs to pathways
    :param pathways_tf_cors:
    :return:
    """
    # Extract pathways and tfs information fro pathways_tf_cors directly
    pathways = {p[0:p.rindex('_')] for p in pathways_tf_cors.columns}
    # Convert to list
    pathways = list(pathways)
    tfs = list(pathways_tf_cors.index)
    # The last is cor. Remove it
    tfs.pop()
    network = nx.DiGraph()
    for pathway in pathways:
        network.add_node(pathway, type='Pathway', color='blue')
    for tf in tfs:
        network.add_node(tf, type='TF', color='cyan')
    for tf in tfs:
        for pathway in pathways:
            pvalue = pathways_tf_cors.loc[tf, pathway + "_pvalue"]
            param = pathways_tf_cors.loc[tf, pathway + "_param"]
            if pvalue < p_value_cutff and math.fabs(param) > param_cutoff:
                color = '#008000' if param > 0 else '#FF0000'
                annotation = 'tf_pathway_activation' if param > 0 else 'tf_pathway_inhibition'
                network.add_edge(tf, pathway, color=color,
                                 value=math.fabs(param), annotation=annotation)  # value used for weight in pyviz
    return network


def simplify_network_for_tfs_pathways(pathway2genes: dict,
                                      tf2genes: dict,
                                      network: nx.DiGraph,
                                      need_parent_pathway: bool = True,
                                      for_pathway_only: bool = False,
                                      use_direct_interaction: bool = True,
                                      p_value_for_direction_interaction: float = 0.05,
                                      add_tf_links: bool = True,
                                      check_with_tf_cor: bool = True,
                                      adata: sc.AnnData = None,
                                      tf_cor_cutoff: float = 0.25,
                                      add_pathway_to_tf_links: bool = True,
                                      delete_pathway_only_component: bool = True,
                                      delete_leaf_pathway: bool = True,
                                      file_output: str = 'test.html',
                                      total_genes: int = 17057) -> (nx.DiGraph, pd.DataFrame):
    """
    Build a network from TFs to pathways by analyzing the covergage and simplifying connections.
    :param use_direct_interaction: true to include directed targetted pathways of TFs based on overlap analysis
    :param p_value_for_direction_interaction: when use_use_direct_interaction is true, use this p_value to selec directed
    interactio
    :param add_pathway_to_tf_links: add links from a pathway to a tf if the pathway contains tf if true.
    :param check_with_tf_cor: true to check correlations for linking tfs
    :param adata: make sure adata exists if check_with_tf_cor is true
    :param delete_pathway_only_component: delete pathway only weakly components if true
    :param delete_leaf_pathway: true to delete pathways that are not impacte by any TF by linked to other pathways.
    :param file_output: to output an interactive html file using pyviz
    :param total_genes: the total genes in the scRNA-seq data.
    """
    # Make sure these condictions are right
    if check_with_tf_cor:
        if adata is None or pa.TF_AUCELL_KEY not in adata.obsm.keys():
            raise ValueError('Make sure adata is passed and {} is in '
                             'adata.obsm since check_with_tf_cor is true!'.format(pa.TF_AUCELL_KEY))
    # As of August 2, 2022, collect all pathways first regardless if they are connected or not
    pathways_in_network = [x for x, y in network.nodes(data=True) if y['type'] == 'Pathway']
    overlap_p_values = pd.DataFrame(columns=['TF', 'Pathway', 'p-Value'])
    row = 0
    # Use the Reactome pathway hierachy for parent/child relationships
    n_network = build_parent_network(pathways_in_network)
    # Assign type and color
    nx.set_node_attributes(n_network, 'Pathway', 'type')
    nx.set_node_attributes(n_network, 'blue', 'color')
    nx.set_edge_attributes(n_network, 'pathway_pathway_hierarchy', 'annotation')
    # Assign color
    if not for_pathway_only:
        # Add TFs
        tfs_in_network = [x for x, y in network.nodes(data=True) if y['type'] == 'TF']
        for tf in tfs_in_network:
            tf_edges = network.out_edges(tf)
            if len(tf_edges) == 0:
                continue
            n_network.add_node(tf, type='TF', color='cyan')
            tf_edge_pathways = [v for u, v in tf_edges]
            # Get the original edges and make sure only one is added in the new network if multiple pathways are hit in
            # the same branch
            # For each pathway, get its ancestors and remove them from the list
            to_be_removed = set()
            for edge_pathway in tf_edge_pathways:
                ancestors = []
                _collect_ancesctor_pathways(n_network, edge_pathway, ancestors)
                to_be_removed.update(ancestors)
            for edge_pathway in tf_edge_pathways:
                if edge_pathway in to_be_removed:
                    # print('Not added: {}'.format(edge_pathway))
                    continue
                if edge_pathway not in n_network:
                    print('Warning: "{}" is not in the Reactome pathway tree.'.format(edge_pathway))
                    continue;
                overlap_p_value = _calculate_overlap_p_value(tf2genes[tf],
                                                             pathway2genes[edge_pathway],
                                                             total_genes)
                overlap_p_values.loc[row] = [tf, edge_pathway, overlap_p_value]
                row += 1
                color_new = network[tf][edge_pathway]['color']
                annotation = network[tf][edge_pathway]['annotation']
                if overlap_p_value < p_value_for_direction_interaction:
                    n_network.add_edge(tf,
                                       edge_pathway,
                                       color=color_new,
                                       annotation=annotation,
                                       value=network[tf][edge_pathway]['value'])  # Value used for weight
                elif not use_direct_interaction:
                    if color_new == '#008000':
                        color_new = '#CCFFCC'
                    else:
                        color_new = '#FFCCCC'  # 90% lighter
                    n_network.add_edge(tf,
                                       edge_pathway,
                                       color=color_new,
                                       annotation=annotation + "_indirect",
                                       value=network[tf][edge_pathway]['value'])  # Value used for weight
    if add_tf_links:
        # Add TFs
        tfs_in_network = [x for x, y in n_network.nodes(data=True) if y['type'] == 'TF']
        build_tfs_network(tfs_in_network, tf2genes, check_with_tf_cor, tf_cor_cutoff, adata, n_network)
    if add_pathway_to_tf_links:
        tfs_in_network = [x for x, y in n_network.nodes(data=True) if y['type'] == 'TF']
        pathways_in_network = [x for x, y in n_network.nodes(data=True) if y['type'] == 'Pathway']
        add_pathway_to_tf_network(pathways_in_network, tfs_in_network, pathway2genes, n_network)
    if delete_pathway_only_component:
        _remove_pathway_only_component(n_network)
    if delete_leaf_pathway:
        _remove_leaf_pathways(n_network)
    # Remove nodes that are not connected
    to_be_removed = []
    for node, atts in n_network.nodes(data=True):
        if n_network.degree(node) == 0:
            to_be_removed.append(node)
        elif not need_parent_pathway:
            # Delete pathway nodes that don't link to any TFs
            if atts['type'] == 'Pathway':
                needed = False
                for neighbor in n_network.predecessors(node):
                    if n_network.nodes[neighbor]['type'] == 'TF':
                        needed = True
                        break
                if not needed:
                    # Check other ways
                    for neighbor in n_network.successors(node):
                        if n_network.nodes[neighbor]['type'] == 'TF':
                            needed = True
                            break
                if not needed:
                    to_be_removed.append(node)
    n_network.remove_nodes_from(to_be_removed)
    if file_output is not None:
        ng = Network(width=1200, height=800, directed=True, notebook=True)
        # ng.toggle_physics(True)
        ng.from_nx(n_network)
        ng.show_buttons(filter_=['physics'])
        ng.show(file_output)
    return n_network, overlap_p_values


def display_network(network, width = 800, height = 800):
    """
    Generate the network for display.
    :param network:
    :return:
    """
    ng = Network(directed=True, notebook=True, width=width, height=height)
    # Toggle physics on to give a better layout
    ng.toggle_physics(True)
    ng.from_nx(network)
    # Somehow the following cannot work. But it should work in the example.html file.
    ng.show_buttons(filter_=['physics'])
    return ng


def add_median_to_nodes(network, df):
    node2att = dict()
    for index, row in df.iterrows():
        att = {'value':row['median_diff']}
        node2att[row['feature']] = att
    nx.set_node_attributes(network, node2att)


def _collect_ancesctor_pathways(network: nx.DiGraph,
                                pathway: str,
                                ancestors: list):
    if pathway is None:
        return
    out_edges = network.out_edges(pathway)
    for u, v in out_edges:
        ancestors.append(v)
        _collect_ancesctor_pathways(network, v, ancestors)


def _calculate_overlap_p_value(genes1: list,
                               genes2: list,
                               total_genes: int):
    """
    Perform a hypergeomic test to calculate p-values for the overlap.
    :param genes1:
    :param genes2:
    :param total_genes:
    :return:
    """
    genes_1 = set(genes1)
    genes_2 = set(genes2)
    overlapped = genes_1 & genes_2
    rv = hypergeom(total_genes, len(genes_1), len(genes_2))
    return 1 - rv.cdf(len(overlapped) - 1)  # Need to counter itself.


def _remove_leaf_pathways(network: nx.DiGraph) -> nx.DiGraph:
    is_changed = True
    while is_changed:
        is_changed = False
        pathways = [n for n, a in network.nodes(data=True) if a['type'] == 'Pathway']
        to_be_removed = []
        for pathway in pathways:
            if network.out_degree(pathway) == 1 and network.in_degree(pathway) == 0:
                for u, v in network.out_edges(pathway):
                    if network.nodes[v]['type'] == 'Pathway':
                        to_be_removed.append(u)
        if len(to_be_removed) > 0:
            is_changed = True
            network.remove_nodes_from(to_be_removed)
    return network


def _remove_pathway_only_component(network: nx.DiGraph) -> nx.DiGraph:
    """
    Delete components that don't have any TFs
    :param network:
    :return:
    """
    to_be_removed = []
    has_tf = False
    for component in nx.weakly_connected_components(network):
        has_tf = False
        for node in component:
            if network.nodes[node]['type'] == 'TF':
                has_tf = True
                break
        if not has_tf:
            to_be_removed.extend(component) # component is a set. Make sure to use extend to unecapsulte elements
    network.remove_nodes_from(to_be_removed)
    return network


def build_parent_network(pathways: list) -> networkx.DiGraph:
    """
    Basically create a simplified pathway hierarchy for the list of passed pathways.
    :param pathways:
    :return:
    """
    reactome_network = load_pathway_hierarchy()
    # To be returned
    pathway_network = networkx.DiGraph(reactome_network)
    selected_nodes = set()
    # Check each pathway in reactome_network and see if it should be included
    for node in reactome_network:
        if include_node(node, reactome_network, pathways):
            selected_nodes.add(node)
    pathway_network.remove_nodes_from(set(pathway_network.nodes) - selected_nodes)
    # Remove nodes that are not in the pathways list and link nodes in the pathways list together
    has_changed = True
    to_be_removed = []
    while has_changed:
        has_changed = False
        to_be_removed.clear()
        # To avoid error
        nodes = list(pathway_network.nodes)
        for node in nodes:
            # Check if this node should be removed
            if node not in pathways:
                to_be_removed.append(node)
                # Re-link: our network links from child pathways to parent pathways
                for parent in pathway_network.successors(node):
                    for child in pathway_network.predecessors(node):
                        pathway_network.add_edge(child, parent)
        if len(to_be_removed) > 0:
            pathway_network.remove_nodes_from(to_be_removed)
            has_changed = True
    return pathway_network


def include_node(node, reactome_network, pathways) -> bool:
    if node in pathways:
        return True
    # If both ancestors and descendents of node are in pathways, return true since
    # this node may be used for connection. Otherwise return false.
    # Check ancesctors first
    if not _include_node(node, reactome_network, pathways, 1):
        return False
    # Check descendents
    if not _include_node(node, reactome_network, pathways, 0):
        return False
    return True


def _include_node(node, reactome_network, pathways, index) -> bool:
    # This should be a width-first search
    if index == 1: # ancestors check
        out_edges = reactome_network.out_edges(node)
        for out_edge in out_edges:
            if out_edge[index] in pathways:
                return True
        # Width first
        for out_edge in out_edges:
            if _include_node(out_edge[1], reactome_network, pathways, index):
                return True # Any ancesctor should be fine
    elif index == 0: # descedent
        in_edges = reactome_network.in_edges(node)
        for in_edge in in_edges:
            if in_edge[index] in pathways:
                return True
        for in_edge in in_edges:
            if _include_node(in_edge[0], reactome_network, pathways, index):
                return True # Any descend should be good
    return False


def load_pathway_hierarchy(
        file_name: str = '../resources/MouseReactomePathways_Hierarchy_Rel_79_122921.xml') -> networkx.DiGraph:
    """
   Load the Reactome mouse pathway hierarchy tree into a network DiaGrah.
   :param file_name:
   :return:
   """
    # Make sure the file is there
    if not os.path.exists(file_name):
        raise ValueError("{} doesn't exist.".format(file_name))
    # The file is an XML file. Do parsing here.from
    graph = networkx.DiGraph()
    tree = et.parse(file_name)
    root = tree.getroot()
    for child in root:
        _load_pathway_hierarchy(child, graph)
    return graph


def _load_pathway_hierarchy(elm, graph):
    if elm.tag == 'Reaction':
        return  # Don't include reaction
    if len(list(elm)) == 0:
        return  # Get to the bottom. Nothing to do
    current_name = elm.attrib['displayName']
    for child in elm:
        if child.tag == 'Reaction':
            continue
        child_name = child.attrib['displayName']
        graph.add_edge(child_name, current_name)
        _load_pathway_hierarchy(child, graph)
