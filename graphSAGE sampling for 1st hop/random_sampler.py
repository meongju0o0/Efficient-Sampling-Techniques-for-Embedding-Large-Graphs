"""PinSAGE sampler & related functions and classes"""
import numpy as np

from torch import ones

from dgl import backend as F
from dgl import to_simple
from dgl import convert
from dgl.sampling.randomwalks import random_walk
from dgl.sampling.neighbor import sample_neighbors
from dgl.base import EID
from dgl import utils


class RandomSampler(object):
    def __init__(self, G, num_traversals, termination_prob,
                 num_random_walks, num_neighbors, metapath=None, weight_column='weights'):
        assert G.device == F.cpu(), "Graph must be on CPU."
        self.G = G
        self.weight_column = weight_column
        self.num_random_walks = num_random_walks
        self.num_neighbors = num_neighbors
        self.num_traversals = num_traversals

        if metapath is None:
            if len(G.ntypes) > 1 or len(G.etypes) > 1:
                raise ValueError('Metapath must be specified if the graph is homogeneous.')
            metapath = [G.canonical_etypes[0]]
        start_ntype = G.to_canonical_etype(metapath[0])[0]
        end_ntype = G.to_canonical_etype(metapath[-1])[-1]
        if start_ntype != end_ntype:
            raise ValueError('The metapath must start and end at the same node type.')
        self.ntype = start_ntype

        self.metapath_hops = len(metapath)
        self.metapath = metapath
        self.full_metapath = metapath * num_traversals
        restart_prob = np.zeros(self.metapath_hops * num_traversals)
        restart_prob[self.metapath_hops::self.metapath_hops] = termination_prob
        self.restart_prob = F.zerocopy_from_numpy(restart_prob)

    # pylint: disable=no-member
    def __call__(self, seed_nodes):
        seed_nodes = utils.prepare_tensor(self.G, seed_nodes, 'seed_nodes')

        seed_nodes = F.repeat(seed_nodes, self.num_random_walks, 0)
        paths, _ = random_walk(
            self.G, seed_nodes, metapath=self.full_metapath, restart_prob=self.restart_prob)
        src = self.G.srcnodes(self.ntype)
        dst = self.G.srcnodes(self.ntype)

        # count the number of visits and pick the K-most frequent neighbors for each node
        neighbor_graph = convert.heterograph(
            {(self.ntype, '_E', self.ntype): (src, dst)},
            {self.ntype: self.G.number_of_nodes(self.ntype)}
        )
        neighbor_graph = to_simple(neighbor_graph, return_counts=self.weight_column)
        counts = ones(neighbor_graph.num_edges())
        neighbor_graph = sample_neighbors(neighbor_graph, seed_nodes, self.num_neighbors)
        selected_counts = F.gather_row(counts, neighbor_graph.edata[EID])
        neighbor_graph.edata[self.weight_column] = selected_counts

        return neighbor_graph