from summit.data import solvent_ds, ucb_ds, DataSet
from summit.domain import Domain, DescriptorsVariable
from summit.initial_design import LatinDesigner
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder

def multiple_lhs_designs(domain, full_ds:DataSet, final_ds: DataSet,
                         seeds: list, fig=None):
    design_indices = [construct_lhs_design(domain, seed) for seed in seeds]
    pc = PCA(n_components=2)
    pcs = pc.fit_transform(final_ds.standardize())
    if fig is  None:
        fig = plt.figure(figsize=(4, 1))
    fig.subplots_adjust(hspace=0.2, wspace=0.2)
    i=1
    for seed, indices in zip(seeds, design_indices):
        ax = fig.add_subplot(len(seeds), 2, i)
        plot_design(ax, pcs, indices, final_ds, seed)
        i+=1

def construct_lhs_design(domain, seed):
    rs = np.random.RandomState(seed)
    lhs = LatinDesigner(domain, random_state=rs)
    experiments = lhs.generate_experiments(10)
    indices = experiments.get_indices('solvent')[:,0]
    return indices


def plot_design(ax, pcs, indices, final_ds, seed):
    le = LabelEncoder() #Label encoder for the different solvent classes
    solvent_classes = final_ds['solvent_class']
    labels = le.fit_transform(solvent_classes)

    markers = ['^', 'P', 'x', 'v', 'p', 'o', 's', '>', 'D', '<', '*']
    for i, solvent_class in enumerate(le.classes_):
        ix = np.where(solvent_classes==solvent_class)[0]
        mask = np.ones(len(ix), dtype=bool)
        selects = []
        for j, select_idx in enumerate(indices):
            select = np.where(ix==select_idx)[0]
            ix = np.delete(ix, select)
            if len(select)> 0:
                selects.append(j)
        ax.scatter(pcs[ix, 0], pcs[ix, 1], c='k', marker=markers[i], label=solvent_class)
        if selects:
            ax.scatter(pcs[selects, 0], pcs[selects, 1], c='r', marker=markers[i], s=100)

    #Plot formatting
#     ax.legend(bbox_to_anchor=(1, 1.0))
    ax.spines['left'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['bottom'].set_position('zero')
    ax.spines['top'].set_color('none')
    ax.set_title(f'Random Seed:{seed}')