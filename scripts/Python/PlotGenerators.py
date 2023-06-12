"""
This script is used to generate some plots
"""
import os
import os.path as path

import numpy as np
import pandas as pd
import scanpy as sc
import scanpy.external as sce
import networkx as nx


# Based on https://stackoverflow.com/questions/51937381/increase-dpi-of-matplotlib-show-in-jupyter-notebook
# import matplotlib as mpl
from statsmodels.tsa.vector_ar.var_model import ma_rep

DPI = 720 # This seems a good number to avoid anything weird in the plot
# mpl.rcParams['figure.dpi'] = DPI
# Need a customized map
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import pyplot as plt

# To be used as marker genes for scoring clusters
MOUSE_CELL_MARKERS = './resources/Mouse_cell_markers.txt'

colors_custom = ['#cfcfcf', "#3300FF"] # Light grey to blue
cmap_custom = LinearSegmentedColormap.from_list("mycmap", colors_custom)
cmap_custom = 'viridis_r'

# Make sure this points to the right file
CELL_TYPE_ANNOTATION_FILE = '/Volumes/ssd/results/missy_sc_rna/bmi1/12_17_cell_annotations.csv'

random_state = 17051256


def _load_mouse_gene_markers(tissue: tuple = ('Intestine', 'Intestinal crypt'),
                             include_sc_markers: bool = False) -> dict:
    markers_pd = pd.read_csv(MOUSE_CELL_MARKERS, sep="\t")
    # Do some selection
    markers_pd = markers_pd[markers_pd['tissueType'].isin(tissue)]
    if not include_sc_markers:
        markers_pd = markers_pd[markers_pd['markerResource'] != 'Single-cell sequencing']
    # Convert it into a dict
    name2markers = {}
    markers_pd.reset_index()
    for index, row in markers_pd.iterrows():
        genes = row['geneSymbol'].split(', ')
        # Remove NA
        genes = [gene for gene in genes if gene != 'NA']
        if not include_sc_markers:
            sub_name = ','.join(genes)
        else:
            sub_name = row['PMID']
        name2markers[row['cellName'] + "(" + sub_name + ")"] = genes
    return name2markers;


def score_markers(adata: sc.AnnData,
                  include_sc_markers: bool = True):
    name2markers = _load_mouse_gene_markers(include_sc_markers=include_sc_markers)
    score_names = []
    for name in name2markers.keys():
        size = len(name2markers[name])
        if size < 50:
            size = 50
        try:
            sc.tl.score_genes(adata, name2markers[name], ctrl_size=size, score_name=name, random_state=random_state)
            score_names.append(name)
        except ValueError:
            print("ValueError for {}.".format(name))
    sc.pl.umap(adata, color = score_names, color_map = cmap_custom)


def plot_genes(anndata: sc.AnnData,
               gene_list,
               out_file_name):
    x_ticks = [-10, 0, 10, 20]
    y_ticks = x_ticks
    out_dir = DIR + "new_figures/"
    from os.path import exists
    if not exists(out_dir):
        os.mkdir(out_dir)
    plot_gene_expressions(anndata, gene_list, out_dir, out_file_name,
                          size=5,
                          x_ticks=x_ticks, y_ticks=y_ticks)
    tokens = out_file_name.split(".")
    # Plot with 17_5 data only
    batch = 'e17_5'
    out_file_name = tokens[0] + "_" + batch + "_." + tokens[1]
    plot_gene_expressions(anndata, gene_list, out_dir, out_file_name,
                          batch=batch,
                          size=5,
                          x_ticks=x_ticks, y_ticks=y_ticks)


def plot_gene_expressions(anndata: sc.AnnData,
                          gene_list,
                          out_dir = ".",
                          save_plot = False,
                          size = 4,
                          cmap = cmap_custom,
                          batch = None,
                          x_ticks = None,
                          y_ticks = None):
    selected_genes = []
    vmin = []
    vmax = []
    all_genes = anndata.var_names
    for gene in gene_list:
        if gene not in all_genes and gene not in anndata.obs.keys():
            print("Cannot find {}.".format(gene))
            continue
        gene_exp = anndata.obs_vector(gene)
        if gene in all_genes:
            vmin.append(gene_exp.min())
            vmax.append(gene_exp.max())
        else:
            vmin.append(None)
            vmax.append(None)
        selected_genes.append(gene)
    _plot_multiple_features(anndata, batch, cmap, out_dir, save_plot,
                            selected_genes, size, vmax, vmin,
                            x_ticks,
                            y_ticks)


def _plot_multiple_features(anndata, batch, cmap, out_dir, save_plot, selected_genes, size, vmax, vmin, x_ticks,
                            y_ticks):
    if batch is not None:
        # Do a filter
        which = anndata.obs['batch'] == batch
        anndata = anndata[which]
    if save_plot is not False and isinstance(save_plot, str):  # Make sure save_plot is a string
        os.chdir(out_dir)
        if x_ticks is None or y_ticks is None:
            sc.pl.umap(anndata,
                       color=selected_genes,
                       size=size,
                       show=False,
                       save=save_plot,
                       color_map=cmap,
                       vmin=vmin,
                       vmax=vmax,
                       wspace=0.20)  # save_plot should be string filename -- with ending, e.g. .pdf
        else:
            umap_plot = sc.pl.umap(anndata,
                                   color=selected_genes,
                                   size=size,
                                   show=False,
                                   color_map=cmap,
                                   vmin=vmin,
                                   vmax=vmax,
                                   wspace=0.20,
                                   legend_loc="on data")  # Control the distance between two plots in the same row
            if isinstance(umap_plot, list):
                for umap_plot_1 in umap_plot:
                    umap_plot_1.set_xticks(x_ticks)
                    umap_plot_1.set_yticks(y_ticks)
            else:
                umap_plot.set_xticks(x_ticks)
                umap_plot.set_yticks(y_ticks)
            # This line code is got from the scanpy's _utils.py.
            plt.savefig(save_plot, dpi=DPI, bbox_inches='tight')
    else:
        sc.pl.umap(anndata,
                   color=selected_genes,
                   size=size,
                   color_map=cmap)


def plot_multiple_pathway_scores(anndata: sc.AnnData,
                                 feature_type,
                                 feature_ids,
                                 out_dir = ".",
                                 save_plot = False,
                                 size = 4,
                                 cmap = cmap_custom,
                                 batch = None,
                                 x_ticks = None,
                                 y_ticks = None):
    """
    This function was copied from Chris' original code for plotting pathway activities
    :param anndata:
    :param feature_type:
    :param feature_ids:
    :param out_dir:
    :param save_plot:
    :param size:
    :param cmap:
    :return:
    """
    # Make a copy to avoid any change to the original anndata
    anndata = anndata.copy()
    selected_pathways = []
    keys = anndata.obsm[feature_type].columns
    vmin = []
    vmax = []
    for feat in feature_ids:
        if feat not in keys:
            print("{} not in {}".format(feat, feature_type))
            continue
        anndata.obs[feat] = anndata.obsm[feature_type].loc[:, feat]
        vmin.append(min(anndata.obs[feat]))
        vmax.append(max(anndata.obs[feat]))
        selected_pathways.append(feat)
    _plot_multiple_features(anndata, batch, cmap, out_dir, save_plot,
                            selected_pathways, size, vmax, vmin,
                            x_ticks,
                            y_ticks)


def plot_pathways(anndata, pathway_list, out_file_name):
    """
    The entry point to plot
    :return:
    """
    feature_name = "X_aucell"
    # Make sure the total features are 16 only
    total = 16
    calls = int(len(pathway_list) / total + 1)
    x_ticks = [-10, 0, 10, 20]
    y_ticks = x_ticks
    out_dir = DIR + "new_figures/"
    from os.path import exists
    if not exists(out_dir):
        os.mkdir(out_dir)
    for i in range(calls):
        start = i * total
        end = start + total
        if end > len(pathway_list):
            end = len(pathway_list)
        pathway_list_1 = pathway_list[start:end]
        tokens = out_file_name.split(".")
        out_file_name_1 = tokens[0] + "_" + str(i) + "." + tokens[1]
        plot_multiple_pathway_scores(anndata,
                                     feature_name,
                                     pathway_list_1,
                                     out_dir,
                                     out_file_name_1,
                                     x_ticks=x_ticks,
                                     y_ticks=y_ticks)
        # Plot with 17_5 data only
        batch = 'e17_5'
        out_file_name_1 = tokens[0] + "_" + batch + "_" + str(i) + "." + tokens[1]
        plot_multiple_pathway_scores(anndata,
                                     feature_name,
                                     pathway_list_1,
                                     out_dir,
                                     out_file_name_1,
                                     batch=batch,
                                     x_ticks=x_ticks,
                                     y_ticks=y_ticks)


def plot_cytotrace(annData: sc.AnnData,
                   cytotrace_file: str,
                   out_dir: str = ".",
                   file_name: str = "umap.pdf",
                   batch: str = None):
    """
    Plot the pre-calculated cytotrace data onto the umap.
    :param annData:
    :param cytotrace_file:
    :return:
    """
    load_cytotrace(annData, cytotrace_file)
    if batch is not None:
        which = annData.obs['batch'] == 'e17_5'
        annData = annData[which]
    os.chdir(out_dir)
    sc.pl.umap(annData,
               color = 'cytotrace',
               size = 4,
               color_map = cmap_custom,
               vmin = 0.0,
               vmax = 1.0,
               show = False,
               save = file_name)


def load_cell_annotations(annData: sc.AnnData,
                          cell_annot_file: str = CELL_TYPE_ANNOTATION_FILE)->None:
    cell_annot_df = pd.read_csv(cell_annot_file)
    cell_annot_df.set_index('cell', inplace=True)
    cell_names = cell_annot_df.index.to_list()
    new_cell_names = _convert_cell_names(cell_names)
    cell_annot_df['new_cell'] = new_cell_names
    cell_annot_df.set_index('new_cell', inplace=True)
    ann_cell_names = annData.obs.index.to_list()
    # This is stupid. There should be a python way to do this. Anyhow:
    cell_types = []
    for cell in ann_cell_names:
        if cell in new_cell_names:
            temp = cell_annot_df.loc[cell, 'annotated_clusters']
            cell_types.append(temp)
        else:
            cell_types.append(None)  # For the time being
            print("Cannot find celltype for {}".format(cell))
    annData.obs['cell_type'] = cell_types


def load_cytotrace(annData: sc.AnnData,
                   cytotrace_file: str):
    cytotrace_df = pd.read_csv(cytotrace_file, delimiter="\t")
    cell_names = cytotrace_df.index.to_list()  # Convert it into a list
    cytotrace_cells = _convert_cell_names(cell_names)
    cytotrace_df['Cell'] = cytotrace_cells
    cytotrace_df.set_index('Cell', inplace=True)
    cell_names = annData.obs.index.to_list()
    # This is stupid. There should be a python way to do this. Anyhow:
    cytotrace_score = []
    for cell in cell_names:
        if cell in cytotrace_cells:
            temp = cytotrace_df.loc[cell, 'CytoTRACE']
            if isinstance(temp, float):
                cytotrace_score.append(temp)
            else:
                cytotrace_score.append(None)
                print(temp)
        else:
            cytotrace_score.append(0.0)  # For the time being
            print("Cannot find score for {}".format(cell))
    annData.obs['cytotrace'] = cytotrace_score


def _convert_cell_names(cell_names):
    new_cell_names = []
    for cell_name in cell_names:
        if cell_name.startswith("17_5_"):
            new_cell_names.append(cell_name.split("_")[2] + "-1-E17.5")
        else:
            new_cell_names.append(cell_name.split("_")[2] + "-1-E12.5")
    return new_cell_names


def get_batch(anndata: sc.AnnData,
              batch: str):
    if 'batch' not in anndata.obs.keys():
        raise ValueError("There is no batch in the passed data.")
    which = anndata.obs['batch'] == batch
    return anndata[which]


# The following functions are modified from code hosted at fi_sc_analysis
def _reset_paga_pos(adata):
    """
    The following code is based on scatterplots.py to calculate paga positions directly from
    the umap. The original paga's pos is not matched with umap because of different layout algorithm
    used.
    :param adata:
    :return: the median coordinates of all cells in individual clusters.
    """
    key = 'leiden'
    umap_key = "X_umap"
    clusters = adata.obs[key].cat.categories
    cluster_pos = np.zeros((len(clusters), 2))
    for icluster, cluster in enumerate(clusters) :
        selected = adata.obs[key] == cluster
        selected_umap = adata.obsm[umap_key][selected, :]
        x_pos, y_pos = np.median(selected_umap, axis=0)
        cluster_pos[icluster] = [x_pos, y_pos]
    return cluster_pos


def run_paga(adata:sc.AnnData):
    sc.pp.neighbors(adata, random_state=random_state)
    sc.tl.paga(adata)
    sc.pl.paga(adata, plot=False, random_state=random_state)
    adata.uns['paga']['pos'] = _reset_paga_pos(adata)


def run_umap_via_paga(adata:sc.AnnData, use_rep = None):
    """
    Follow the procedure outlined in https://scanpy-tutorials.readthedocs.io/en/latest/pbmc3k.html
    by running paga first as the init_pos.
    :param adata: assumed this data has been cleaned and normalized.
    :param use_rep: for the harmony batch corrected data, X_pca_harmony should be used.
    :return:
    """
    if use_rep is None:
        sc.pp.highly_variable_genes(adata)
        sc.pp.pca(adata, random_state=random_state)
        use_rep = 'X_pca'
    sc.pp.neighbors(adata, use_rep=use_rep, random_state=random_state)
    sc.tl.leiden(adata, random_state=random_state)
    sc.tl.paga(adata)
    sc.pl.paga(adata, plot=False, random_state=random_state)
    sc.tl.umap(adata, random_state=random_state, init_pos='paga')
    adata.uns['paga']['pos'] = _reset_paga_pos(adata)


def plot_paga(anndata: sc.AnnData):
    # This is the root cell selected based on the maximum cytotrace value
    # This cell is also used for dpt inference
    root_cell = 'GTGCACGCACCCAAGC-1-1-e17_5'
    root_index = np.flatnonzero(anndata.obs_names == root_cell)[0]
    sc.pl.paga(anndata,
               threshold=0.45,
               color = ['leiden', 'cytotrace', 'dpt_pseudotime'],
               pos = anndata.uns['paga']['pos'],
               cmap = cmap_custom,
               root = root_index) # Don't make any difference by assigning root


def run_dpt(adata:sc.AnnData,
            root_cell: str):
    # Get the number index of the root_cell
    root_index = np.flatnonzero(adata.obs_names == root_cell)[0]
    adata.uns['iroot'] = root_index
    sc.tl.dpt(adata)


def open_data(file: str,
              preprocess: bool = True) -> sc.AnnData:
    """
    Open data the data and then pre-process it
    :param file:
    :param preprocess: true for preprocess
    :return:
    """
    adata = None
    if path.isdir(file):
        adata = sc.read_10x_mtx(file, var_names='gene_symbols', cache=True)
    else: # This should be a file. Assume it is a h5ad
        adata = sc.read_h5ad(file)
    if adata is None:
        raise ValueError('Cannot open file {}.'.format(str))
    if not preprocess:
        return adata
    # Copy from the scanpy_wrapper and from Chris' code
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


def run_umap(adata,
             need_run_variable: bool=False):
    if need_run_variable:
        sc.pp.highly_variable_genes(adata)
    sc.pp.pca(adata, random_state=random_state)
    sc.pp.neighbors(adata, random_state=random_state)
    sc.tl.leiden(adata, random_state=random_state)
    sc.tl.umap(adata, random_state=random_state)
    return adata


def load_e12_data():
    file = '/Users/wug/Documents/missy_single_cell/seq_data_v2/12_5_gfp/filtered_feature_bc_matrix'
    return open_data(file)


def load_e17_data():
    file = '/Users/wug/Documents/missy_single_cell/seq_data_v2/17_5_gfp/filtered_feature_bc_matrix'
    return open_data(file)

def ingest(source: sc.AnnData,
           reference: sc.AnnData) -> sc.AnnData:
    obs = None
    if 'leiden' in reference.obs.keys():
        obs = 'leiden'
    reference = reference.copy()
    var_names = reference.var_names.intersection(source.var_names)
    adata_ref = reference[:, var_names]
    adata = source[:, var_names]
    sc.tl.ingest(adata, adata_ref, obs=obs)
    return adata


def merge(source: sc.AnnData,
          target: sc.AnnData,
          batch_categories: str = None,
          batch_correction: str = None):
    """
    Merge the source dataset into the target dataset
    :param source:
    :param target:
    :param batch_categories:
    :param batch_correction:
    :return:
    """
    merged = source.concatenate(target, batch_categories=batch_categories)
    if batch_correction is not None:
        # sc.pp.highly_variable_genes(merged)
        sc.pp.pca(merged) # Need to conduct pca first for both batch correction
        if batch_correction == 'harmony':
            sce.pp.harmony_integrate(merged, 'batch')
        elif batch_correction == 'bbknn':
            # See https://github.com/theislab/scanpy/issues/514
            import bbknn
            bbknn.bbknn(merged, 'batch')
        else:
            raise ValueError("{} is not supported in batch correction.".format(batch_correction))
    # Want to split them into the original data for doing whatever it is needed
    if batch_categories is None:
        batch_categories = ['0', '1']
    src_data = merged[merged.obs['batch'] == batch_categories[0]]
    target_data = merged[merged.obs['batch'] == batch_categories[1]]
    return src_data, target_data, merged


def calculate_overlap_via_neighbors(adata: sc.AnnData):
    """
    Calculate the overlap of two datasetes in the passed merged data based on neighbors.
    Make sure connectivities is in the obsp keys.
    :param adata:
    :return:
    """
    neighbor_key = 'connectivities'
    if neighbor_key not in adata.obsp.keys():
        raise ValueError("{} not in obsp.".format(neighbor_key))
    # For easy handling, generate a network object
    network = nx.Graph(adata.obsp[neighbor_key])
    overlap_counter = 0
    total_nodes = 0
    for n in network:
        n_batch = adata.obs['batch'][n]
        total_nodes += 1
        # Get it neighbor
        neigbors = network.neighbors(n)
        isOverlapped = False
        for nb in neigbors:
            nb_batch = adata.obs['batch'][nb]
            if n_batch != nb_batch:
                overlap_counter += 1
                break
    percentage = overlap_counter / total_nodes
    print("Total overlapped: {} in total {}: {}".format(overlap_counter, total_nodes, percentage))



# Some simple configuration
DIR = "/Volumes/ssd/results/missy_sc_rna/bmi1/"
# This file was generated by Chris with 12.5 and 17.5 files merged
file_name = DIR + "merged2.h5ad"
batch_categories = ['E12.5', 'E17.5']
# Load the anndata
# sc.pl.umap(anndata, color = 'leiden')
# anndata = get_batch(anndata, "e17_5")
# run_paga(anndata)

cytotrace_file = '/Volumes/ssd/results/missy_sc_rna/nasim_sc_out_rws00061/E12E17/CytoTRACE_plot_table_E12E17.txt'
celltype_file = '/Volumes/ssd/results/missy_sc_rna/bmi1/12_17_cell_annotations.csv'
out_file_name = "_cytotrace_E12_E17_5.pdf"
# plot_cytotrace(anndata, cytotrace_file, DIR, out_file_name)
out_file_name = "_cytotrace_E17_5.pdf"
# plot_cytotrace(anndata, cytotrace_file, DIR, out_file_name, batch="e17_5")
# anndata = sc.read_h5ad(file_name)
# load_cytotrace(anndata, cytotrace_file)
# load_cell_annotations(anndata, celltype_file)


# Save plot
# These files are collected from Missy's file list, joint_pathway_anova_results-selected for umap 11.08.2021
# Green list for interesting pathways
pathway_list = ['Digestion of dietary lipid',
                'Cobalamin (Cbl, vitamin B12) transport and metabolism',
                'CLEC7A/inflammasome pathway',
                'Plasma lipoprotein assembly',
                'Fatty acid metabolism',
                'Downregulation of ERBB2 signaling',
                'Downregulation of ERBB4 signaling',
                'Oncogene Induced Senescence',
                'Regulation of FZD by ubiquitination']
out_file_name = "umap_pathway_scores_green_110921.pdf"

# Blue pathways for differentiation
# pathway_list = [
# 'Metabolism of water-soluble vitamins and cofactors',
# 'Glucagon-type ligand receptors',
# 'Ion channel transport',
# 'Peptide hormone metabolism', # Cannot find this pathway in the AUCELL data. Different version?
# 'Cellular responses to stress',
# 'Vesicle-mediated transport', # Not in the pathway score file
# 'NOTCH3 Activation and Transmission of Signal to the Nucleus',
# 'Signaling by NOTCH3',
# 'Iron uptake and transport',
# 'Activated NOTCH1 Transmits Signal to the Nucleus',
# 'Signaling by NOTCH',
# 'Cytokine Signaling in Immune system',
# 'DEx/H-box helicases activate type I IFN and inflammatory cytokines production',
# 'Growth hormone receptor signaling'
# ]
# out_file_name = "umap_pathway_scores_blue_110921.pdf"

# # # Yellow pathways for stem
# pathway_list = [
# 'Cell Cycle',
# 'Cell Cycle, Mitotic',
# 'M Phase',
# 'Regulation of PLK1 Activity at G2/M Transition',
# 'Mitotic Metaphase and Anaphase',
# 'Mitotic Anaphase',
# 'G2/M Transition',
# 'Mitotic G2-G2/M phases',
# 'Chromatin modifying enzymes',
# 'Chromatin organization',
# 'HuR (ELAVL1) binds and stabilizes mRNA',
# 'SLBP independent Processing of Histone Pre-mRNAs',
# 'SLBP Dependent Processing of Replication-Dependent Histone Pre-mRNAs',
# 'G2/M Checkpoints',
# 'Regulation of localization of FOXO transcription factors',
# 'Transcriptional regulation by RUNX2',
# 'FOXO-mediated transcription',
# 'Glycolysis',
# 'Beta-catenin independent WNT signaling',
# 'Glucose metabolism',
# 'Regulation of RUNX2 expression and activity',
# 'TGF-beta receptor signaling activates SMADs',
# 'EGFR downregulation',
# 'Signaling by ERBB2',
# 'Regulation of RUNX3 expression and activity',
# 'RUNX1 interacts with co-factors whose precise effect on RUNX1 targets is not known',
# 'TCF dependent signaling in response to WNT',
# 'RUNX1 regulates transcription of genes involved in differentiation of HSCs',
# 'Transcriptional regulation by RUNX3',
# 'Signaling by WNT',
# 'Signaling by EGFR',
# 'SMAD2/SMAD3:SMAD4 heterotrimer regulates transcription',
# 'Downregulation of SMAD2/3:SMAD4 transcriptional activity',
# 'Transcriptional activity of SMAD2/SMAD3:SMAD4 heterotrimer',
# 'WNT5A-dependent internalization of FZD4',
# 'WNT5A-dependent internalization of FZD2, FZD5 and ROR2',
# 'HDMs demethylate histones',
# 'WNT ligand biogenesis and trafficking',
# 'Repression of WNT target genes',
# 'Signaling by Hippo',
# 'Signaling by NODAL',
# 'RUNX3 regulates NOTCH signaling',
# 'RUNX3 regulates WNT signaling',
# 'Regulation of RUNX1 Expression and Activity',
# 'Regulation of FOXO transcriptional activity by acetylation',
# 'RUNX3 regulates YAP1-mediated transcription',
# 'WNT mediated activation of DVL',
# 'RUNX3 regulates CDKN1A transcription',
# 'SHC1 events in ERBB4 signaling',
# 'GRB2 events in EGFR signaling',
# 'FOXO-mediated transcription of cell cycle genes',
# 'AKT-mediated inactivation of FOXO1A',
# 'FOXO-mediated transcription of oxidative stress, metabolic and neuronal genes'
# ]
# out_file_name = "umap_pathway_scores_yellow_110921.pdf"

# plot_pathways(anndata, pathway_list, out_file_name)

# Plot genes
# gene_list = [
#     'cell_type',
#     'batch',
#     "Pou5f1",
#     "Alpi",
#     "Defa24",
#     "Chga",
#     "Wnt5a",
#     "Wnt3",
#     "Sfrp1",
#     "Sfrp5",
#     "Dkk3"
# ]
# out_file_name = "umap_genes_figure_4f_111121.pdf"
# plot_genes(anndata, gene_list, out_file_name)

# adata_12 = load_e12_data()
# run_umap(adata_12)
# adata_17 = load_e17_data()
# merged_adata = merge(adata_12, adata_17, ['E12', 'E17'], 'harmony')


