import pandas
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec

## load data
nrn = pandas.read_csv('Data//neurons.csv')
con = pandas.read_csv('Data//connections.csv')
coord = pandas.read_csv('Data//coordinates.csv')

## create useful variables
N_neurons = len(nrn)
N_synapses = len(con)
neuropils = np.unique(con['neuropil'])
N_neuropils = len(neuropils)
nt_types = np.array(['ACH', 'GLUT', 'GABA', 'DA', 'SER', 'OCT'])
N_types = len(nt_types)
id = nrn['root_id']
id_to_num = {id[idx]:idx for idx in range(1, N_neurons)}
nt_colors = ['xkcd:coral', 'xkcd:orange', 'xkcd:gold', 'xkcd:light green', 'xkcd:sky blue', 'xkcd:medium purple']

## add neuron's neurotransmitter type to con table
nt_type = nrn['nt_type']
pre_nt_type = [nt_type[id_to_num[con['pre_root_id'][idx]]] for idx in range(N_synapses)]
post_nt_type = [nt_type[id_to_num[con['post_root_id'][idx]]] for idx in range(N_synapses)]
con['pre_nt_type'] = pre_nt_type
con['post_nt_type'] = post_nt_type

## proportion of neurotransmitters in different neuropils
neuropil__type = np.zeros((N_neuropils, N_types))
for idx1 in range(N_neuropils):
    for idx2 in range(N_types):
        neuropil__type[idx1, idx2] = \
            (con['nt_type'][con['neuropil'] == neuropils[idx1]] == nt_types[idx2]).sum()
neuropil__type = neuropil__type / np.tile(neuropil__type.sum(axis=1), (N_types, 1)).T
# plot
fig = plt.figure(constrained_layout=True)
gs = GridSpec(3, 3, figure=fig)
fig.add_subplot(gs[0, :])
plt.imshow(neuropil__type.T, aspect='auto', cmap='Greys', vmin=0, vmax=1), plt.colorbar()
plt.title('Fraction of synapses')
plt.yticks(range(6), nt_types), plt.xticks(range(0, N_neuropils, 2), neuropils[range(0, N_neuropils, 2)])
plt.xticks(rotation=60)
for idx in range(N_types):
    fig.add_subplot(gs[1+int(idx/3), np.mod(idx, 3)])
    frac = np.sort(neuropil__type[:, idx])[-5:][::-1]
    locs = neuropils[np.argsort(neuropil__type[:, idx])[-5:][::-1]]
    plt.bar(range(5), frac, color=nt_colors[idx], alpha=.8, edgecolor='k')
    plt.xticks([]), plt.title(nt_types[idx]+'-rich Neuropils')
    for x, t in zip(range(5), locs):
        plt.text(x, frac.max()/10, t, rotation=90)

## probability of dale's law violations for each type of synapse across neuropils
p_nondale = np.zeros((N_neuropils, N_types))
for idx1 in range(N_neuropils):
    for idx2 in range(N_types):
        nt_type_synapse = con['nt_type'][con['neuropil'] == neuropils[idx1]] == nt_types[idx2]
        nt_type_neuron = con['pre_nt_type'][con['neuropil'] == neuropils[idx1]] == nt_types[idx2]
        p_nondale[idx1, idx2] = 1 - (nt_type_synapse == nt_type_neuron).sum() / len(nt_type_synapse)
# plot
fig = plt.figure(constrained_layout=True)
gs = GridSpec(3, 3, figure=fig)
fig.add_subplot(gs[0, :])
plt.imshow(p_nondale.T, aspect='auto', cmap='Greys', vmin=0, vmax=.2), plt.colorbar()
plt.title('Fraction of synapses that violate Dale\'s law')
plt.yticks(range(6), nt_types), plt.xticks(range(0, N_neuropils, 2), neuropils[range(0, N_neuropils, 2)])
plt.xticks(rotation=60)
for idx in range(N_types):
    fig.add_subplot(gs[1+int(idx/3), np.mod(idx, 3)])
    frac = np.sort(p_nondale[:, idx])[-5:][::-1]
    locs = neuropils[np.argsort(p_nondale[:, idx])[-5:][::-1]]
    plt.bar(range(5), frac, color=nt_colors[idx], alpha=.8, edgecolor='k')
    plt.xticks([]), plt.title(nt_types[idx]+' violations')
    for x, t in zip(range(5), locs):
        plt.text(x, frac.max()/10, t, rotation=90)

## fraction of outgoing connections from different neurotransmitters across neuropils
neuropil__posttype__pretype = np.zeros((N_neuropils, N_types, N_types))
for idx1 in range(N_neuropils):
    for idx2 in range(N_types):
        post = con['post_nt_type'][con['neuropil'] == neuropils[idx1]] == nt_types[idx2]
        for idx3 in range(N_types):
            pre = con['pre_nt_type'][con['neuropil'] == neuropils[idx1]] == nt_types[idx3]
            neuropil__posttype__pretype[idx1, idx2, idx3] = ((post & pre).sum() ** 2) / (post.sum() * pre.sum())
# plot
plt.figure()
for idx in range(77):
    ax = plt.subplot(7, 11, idx+1)
    plt.imshow(neuropil__posttype__pretype[idx], cmap='Greys')
    plt.text(0, 5, neuropils[idx])
    if idx == 0:
        plt.xticks(range(6), nt_types, rotation=90), plt.yticks(range(6), nt_types)
        ax.xaxis.tick_top()
    else:
        plt.xticks([]), plt.yticks([])
plt.show()