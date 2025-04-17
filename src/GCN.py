import os

import torch
import torch_geometric
import gzip
import pickle
import numpy as np
import time

def inverse_join_literals(tensor):
    n, two_k = tensor.shape
    if two_k % 2 != 0:
        raise ValueError("The second dimension must be even.")
    
    k = two_k // 2
    
    # Reshape to (n, k, 2)
    tensor = tensor.view(n, k, 2)
    
    # Permute to (n, 2, k)
    tensor = tensor.permute(0, 2, 1)
    
    # Reshape to (2n, k)
    tensor = tensor.reshape(2 * n, k)
    
    return tensor

def join_literals(tensor):
    two_n, k = tensor.shape
    if two_n % 2 != 0:
        raise ValueError("The first dimension must be even.")

    n = two_n // 2

    # Reshape to (n, 2, k)
    tensor = tensor.view(n, 2, k)

    # Permute to (n, k, 2)
    tensor = tensor.permute(0, 2, 1)

    # Reshape to (n, 2k)
    tensor = tensor.reshape(n, 2 * k)
    
    return tensor
class BackboneAndValuesPredictor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        emb_size = 64
        cons_nfeats = 4
        edge_nfeats = 1
        var_nfeats = 6

        # CONSTRAINT EMBEDDING
        self.cons_embedding = torch.nn.Sequential(
            torch.nn.LayerNorm(cons_nfeats),
            torch.nn.Linear(cons_nfeats, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.ReLU(),
        )

        # EDGE EMBEDDING
        self.edge_embedding = torch.nn.Sequential(
            torch.nn.LayerNorm(edge_nfeats),
        )

        # VARIABLE EMBEDDING
        self.var_embedding = torch.nn.Sequential(
            torch.nn.LayerNorm(var_nfeats),
            torch.nn.Linear(var_nfeats, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.ReLU(),
        )
        
        self.conv_v_to_c = BipartiteGraphConvolution()
        self.conv_c_to_v = BipartiteGraphConvolution()

        self.l_to_l = LiteralsMessage(emb_size)

        self.conv_v_to_c2 = BipartiteGraphConvolution()
        self.conv_c_to_v2 = BipartiteGraphConvolution()

        self.l_to_l2 = LiteralsMessage(emb_size)

        self.output_module = torch.nn.Sequential(
            torch.nn.Linear(2*emb_size, 2*emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(2*emb_size, 3, bias=False),
        )

    def forward(
        self, constraint_features, edge_indices, edge_features, variable_features
    ):
        reversed_edge_indices = torch.stack([edge_indices[1], edge_indices[0]], dim=0)

        # First step: linear embedding layers to a common dimension (64)
        constraint_features = self.cons_embedding(constraint_features)
        edge_features = self.edge_embedding(edge_features)
        variable_features = self.var_embedding(variable_features)

        # Two half convolutions
        constraint_features = self.conv_v_to_c(variable_features, reversed_edge_indices, edge_features, constraint_features)
        variable_features = self.conv_c_to_v(constraint_features, edge_indices, edge_features, variable_features)

        variable_features = self.l_to_l(variable_features)

        constraint_features = self.conv_v_to_c2(variable_features, reversed_edge_indices, edge_features, constraint_features)
        variable_features = self.conv_c_to_v2(constraint_features, edge_indices, edge_features, variable_features)

        variable_features = self.l_to_l2(variable_features)
        variable_features = join_literals(variable_features)
        # A final MLP on the variable features
        output = self.output_module(variable_features).squeeze(-1)
        return output
    
class MilpBackboneAndValuesPredictor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        emb_size = 64
        cons_nfeats = 4
        edge_nfeats = 1
        var_nfeats = 6

        # CONSTRAINT EMBEDDING
        self.cons_embedding = torch.nn.Sequential(
            torch.nn.LayerNorm(cons_nfeats),
            torch.nn.Linear(cons_nfeats, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.ReLU(),
        )

        # EDGE EMBEDDING
        self.edge_embedding = torch.nn.Sequential(
            torch.nn.LayerNorm(edge_nfeats),
        )

        # VARIABLE EMBEDDING
        self.var_embedding = torch.nn.Sequential(
            torch.nn.LayerNorm(var_nfeats),
            torch.nn.Linear(var_nfeats, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.ReLU(),
        )
        
        self.conv_v_to_c = BipartiteGraphConvolution()
        self.conv_c_to_v = BipartiteGraphConvolution()

        self.conv_v_to_c2 = BipartiteGraphConvolution()
        self.conv_c_to_v2 = BipartiteGraphConvolution()

        self.output_module = torch.nn.Sequential(
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, 3, bias=False),
        )

    def forward(
        self, constraint_features, edge_indices, edge_features, variable_features
    ):
        reversed_edge_indices = torch.stack([edge_indices[1], edge_indices[0]], dim=0)

        # First step: linear embedding layers to a common dimension (64)
        constraint_features = self.cons_embedding(constraint_features)
        edge_features = self.edge_embedding(edge_features)
        variable_features = self.var_embedding(variable_features)

        # Two half convolutions
        constraint_features = self.conv_v_to_c(variable_features, reversed_edge_indices, edge_features, constraint_features)
        variable_features = self.conv_c_to_v(constraint_features, edge_indices, edge_features, variable_features)

        constraint_features = self.conv_v_to_c2(variable_features, reversed_edge_indices, edge_features, constraint_features)
        variable_features = self.conv_c_to_v2(constraint_features, edge_indices, edge_features, variable_features)

        # A final MLP on the variable features
        output = self.output_module(variable_features).squeeze(-1)
        return output
    
def flip(x, dim=0):
    # Move the specified dimension to the front
    x = x.transpose(0, dim)
    
    # Get the shape of the tensor
    shape = x.shape
    
    # Ensure the leading dimension is even
    if shape[0] % 2 != 0:
        raise ValueError("The size of the specified dimension must be even.")
    
    # Reshape to collapse the leading dimension to pairs
    new_shape = (shape[0] // 2, 2) + shape[1:]
    x = x.view(new_shape)
    
    # Initialize an empty tensor with the same shape as the reshaped tensor
    result = torch.empty_like(x)
    
    # Perform the swapping for related pairs
    result[:, 0] = x[:, 1]
    result[:, 1] = x[:, 0]
    
    # Reshape back to the original shape
    result = result.view(shape)
    
    # Move the dimensions back to their original order
    result = result.transpose(0, dim)
    
    return result
class LiteralsMessage(torch.nn.Module):
    def __init__(self,size):
        super().__init__()
        self.a = torch.nn.Parameter(torch.randn(size))
        self.b = torch.nn.Parameter(torch.randn(size))
        self.relu = torch.nn.ReLU()
    def forward(self,literals):
        return self.relu(self.a*literals + self.b*flip(literals))
    
class LiteralsMessageFull(torch.nn.Module):
    def __init__(self,size):
        super().__init__()
        self.output = torch.nn.Sequential(
            torch.nn.Linear(2*size, 2*size),
            torch.nn.ReLU())
    def forward(self,literals):
        return inverse_join_literals(self.output(join_literals(literals)))
class BipartiteGraphConvolution(torch_geometric.nn.MessagePassing):
    """
    The bipartite graph convolution is already provided by pytorch geometric and we merely need
    to provide the exact form of the messages being passed.
    """

    def __init__(self,emb_size = 64):
        super().__init__("add")
        

        self.feature_module_left = torch.nn.Sequential(
            torch.nn.Linear(emb_size, emb_size)
        )
        self.feature_module_edge = torch.nn.Sequential(
            torch.nn.Linear(1, emb_size, bias=False)
        )
        self.feature_module_right = torch.nn.Sequential(
            torch.nn.Linear(emb_size, emb_size, bias=False)
        )
        self.feature_module_final = torch.nn.Sequential(
            torch.nn.LayerNorm(emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
        )

        self.post_conv_module = torch.nn.Sequential(torch.nn.LayerNorm(emb_size))

        # output_layers
        self.output_module = torch.nn.Sequential(
            torch.nn.Linear(2 * emb_size, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
        )

    def forward(self, left_features, edge_indices, edge_features, right_features):
        """
        This method sends the messages, computed in the message method.
        """


        output = self.propagate(
            edge_indices,
            size=(left_features.shape[0], right_features.shape[0]),
            node_features=(left_features, right_features),
            edge_features=edge_features,
        )
        #b=torch.cat([self.post_conv_module(output), right_features], dim=-1)
        #a=self.output_module(torch.cat([self.post_conv_module(output), right_features], dim=-1))

        return self.output_module(
            torch.cat([self.post_conv_module(output), right_features], dim=-1)
        )


    def message(self, node_features_i, node_features_j, edge_features):
        #node_features_i,the node to be aggregated
        #node_features_j,the neighbors of the node i

        # print("node_features_i:",node_features_i.shape)
        # print("node_features_j",node_features_j.shape)
        # print("edge_features:",edge_features.shape)

        output = self.feature_module_final(
            self.feature_module_left(node_features_i)
            + self.feature_module_edge(edge_features)
            + self.feature_module_right(node_features_j)
        )

        return output

class GraphDatasetTriOutput(torch_geometric.data.Dataset):
    """
    This class encodes a collection of graphs, as well as a method to load such graphs from the disk.
    It can be used in turn by the data loaders provided by pytorch geometric.
    """

    def __init__(self, sample_files: list):
        super().__init__(root=None, transform=None, pre_transform=None)
        self.sample_files = sample_files

    def len(self):
        return len(self.sample_files)


    def process_sample(self,filepath):
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        BG = data['X']
        sols = data['y']
        return BG,sols

    def get_graph_components(A, v_nodes, c_nodes):
        constraint_features = c_nodes
        edge_indices = A._indices()

        variable_features = v_nodes
        edge_features =A._values().unsqueeze(1)
        edge_features=torch.ones(edge_features.shape)
        constraint_features = torch.nan_to_num(constraint_features,1)
        
        return constraint_features, edge_indices, edge_features, variable_features

    def get(self, index):
        """
        This method loads a node bipartite graph observation as saved on the disk during data collection.
        """

        # nbp, sols, objs, varInds, varNames = self.process_sample(self.sample_files[index])
        BG, sols = self.process_sample(self.sample_files[index])
        A, v_map, v_nodes, c_nodes = BG
        constraint_features, edge_indices, edge_features, variable_features = GraphDatasetTriOutput.get_graph_components(A, v_nodes, c_nodes)    

        graph = BipartiteNodeData(
            torch.FloatTensor(constraint_features.cpu()),
            torch.LongTensor(edge_indices.cpu()),
            torch.FloatTensor(edge_features.cpu()),
            torch.FloatTensor(variable_features.cpu()),
        )
        
        
        
        # We must tell pytorch geometric how many nodes there are, for indexing purposes
        graph.num_nodes = constraint_features.shape[0] + variable_features.shape[0]
        graph.y = torch.FloatTensor(sols)
        l1 = graph.y[::2]>.5
        l0 = graph.y[1::2]>.5
        is_not_backbone = ~(l1 | l0)
        #concatenate l0,l1,is_not_backbone
        graph.y = l1.long()+ is_not_backbone.long() * 2
        graph.ntvars = variable_features.shape[0]
        graph.varNames = [v for _,v in sorted(v_map.items(), key=lambda item: item[1])]

        return graph

class BipartiteNodeData(torch_geometric.data.Data):
    """
    This class encode a node bipartite graph observation as returned by the `ecole.observation.NodeBipartite`
    observation function in a format understood by the pytorch geometric data handlers.
    """

    def __init__(
            self,
            constraint_features,
            edge_indices,
            edge_features,
            variable_features,

    ):
        super().__init__()
        self.constraint_features = constraint_features
        self.edge_index = edge_indices
        self.edge_attr = edge_features
        self.variable_features = variable_features



    def __inc__(self, key, value, store, *args, **kwargs):
        """
        We overload the pytorch geometric method that tells how to increment indices when concatenating graphs
        for those entries (edge index, candidates) for which this is not obvious.
        """
        if key == "edge_index":
            return torch.tensor(
                [[self.constraint_features.size(0)], [self.variable_features.size(0)]]
            )
        elif key == "candidates":
            return self.variable_features.size(0)
        else:
            return super().__inc__(key, value, *args, **kwargs)

