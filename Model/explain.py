import torch
from graph_generation import generate_graph
from rdkit import Chem
from rdkit import Chem, DataStructs
from IPython.display import display, SVG
from rdkit import Chem
import matplotlib.cm as cm
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D,  MolToFile
from dgl.readout import sum_nodes
from torch.utils.data import DataLoader
from rdkit import DataStructs
from rdkit import Chem
import dgl
from torch.utils.data import Dataset
import torch
import torch.optim
import torch.nn as nn
from torch.utils.data import DataLoader
from dgl.data.utils import load_graphs
import dgl
from dgl.nn.pytorch import GINConv
from dgl.readout import sum_nodes

class pred_data(Dataset):
    def __init__(self, graph, smiles):
        self.smiles = smiles
        self.graph = graph
        self.lens = len(smiles)

    def __len__(self):
        return self.lens

    def __getitem__(self, item):
        return self.smiles[item], self.graph[item]

def collate(sample):
    graphs, labels = zip(*sample)
    batched_graph = dgl.batch(graphs)
    batched_graph.set_n_initializer(dgl.init.zero_initializer)
    batched_graph.set_e_initializer(dgl.init.zero_initializer)
    labels = torch.stack(labels, dim=0)
    return batched_graph, labels

def explain(smiles):
    ls_smi = []
    if isinstance(smiles, list):
        ls_smi = smiles
    else:
        ls_smi.append(smiles)
    graph = generate_graph(ls_smi)
    data = pred_data(graph=graph, smiles=ls_smi)
    data_loader = DataLoader(data, batch_size=1, shuffle=False, collate_fn=collate)
    dataset = data_loader.dataset

    model = torch.load('./Model/mul_task.pth')[
        'model_state_dict'].to('cpu')
    smiles, graph = dataset[0]
    bg = dgl.batch([graph])
    atoms_feat = bg.ndata['h']
    atom_weight = model(bg, atoms_feat)[1]
    atom_weights = atom_weight.cpu().detach().numpy()
    score = model(bg, atoms_feat)[0]
    score = score.cpu().detach().numpy()

    ls = []
    for i in list(range(0, len(atom_weights))):
        ls.append(abs(atom_weights[i][0]))
    min_value = min(ls)
    max_value = max(ls)
    weights = (ls - min_value) / (max_value - min_value)
    mol = Chem.MolFromSmiles(smiles)
    norm = cm.colors.Normalize(vmin=0, vmax=1.00)
    cmap = cm.get_cmap('OrRd')
    plt_colors = cm.ScalarMappable(norm=norm, cmap=cmap)

    atom_colors = {i: plt_colors.to_rgba(weights[i]) for i in range(bg.number_of_nodes())}
    rdDepictor.Compute2DCoords(mol)
    dr = rdMolDraw2D.MolDraw2DSVG(400, 370)
    do = rdMolDraw2D.MolDrawOptions()
    do.bondLineWidth = 4
    do.fixedBondLength = 30
    do.highlightRadius = 4
    dr.SetFontSize(1)
    dr.drawOptions().addAtomIndices = True
    mol = rdMolDraw2D.PrepareMolForDrawing(mol)
    dr.DrawMolecule(mol, highlightAtoms=range(bg.number_of_nodes()),
                    highlightBonds=[],
                    highlightAtomColors=atom_colors)
    dr.FinishDrawing()
    svg = dr.GetDrawingText()
    svg = svg.replace('svg:', '')
    return svg 



class WeightAndSum(nn.Module):
    def __init__(self, in_feats):
        super(WeightAndSum, self).__init__()
        self.in_feats = in_feats
        self.atom_weighting = nn.Sequential(
            nn.Linear(in_feats, 1)
        )

    def forward(self, g, feats , get_node_weight=True):
        with g.local_scope():
            g.ndata['h'] = feats
            atom_weights = self.atom_weighting(g.ndata['h'])
            g.ndata['w'] = torch.nn.Sigmoid()(self.atom_weighting(g.ndata['h']))
            h_g_sum = sum_nodes(g, 'h', 'w')
        if get_node_weight:
            return h_g_sum, atom_weights
        else:
            return h_g_sum

class Classifer(nn.Module):
    def __init__(self,
                 graph_hidden,
                 depth,
                 mlp_layers,
                 dropout,
                 ):
        super(Classifer, self).__init__()

        ##第一层GIN
        self.gcn1 = GINConv(apply_func=nn.Linear(38, graph_hidden), aggregator_type="sum")
        self.gcn1_bn = torch.nn.BatchNorm1d(graph_hidden, eps=1e-05, momentum=0.1, affine=True,
                                            track_running_stats=True)
        ##第k层GIN
        self.gcn2 = nn.ModuleList(
            [GINConv(apply_func=nn.Linear(graph_hidden, graph_hidden), aggregator_type="sum") for _ in
             range(depth - 1)])  #
        self.gcn2_bn = nn.ModuleList(
            [torch.nn.BatchNorm1d(graph_hidden, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True) for i in
             range(depth - 1)])
        # 给每个原子加注意力机制
        self.readout = WeightAndSum(graph_hidden)

        # 第一层FC
        self.fc1 = nn.Linear(graph_hidden, mlp_layers[0])
        self.bn1 = torch.nn.BatchNorm1d(mlp_layers[0], eps=1e-05, momentum=0.1,
                                        affine=True, track_running_stats=True)
        # 第k层FC
        self.linears = nn.ModuleList(
            [nn.Linear(mlp_layers[i], mlp_layers[i + 1]) for i in range(len(mlp_layers) - 1)])
        self.bns = nn.ModuleList([torch.nn.BatchNorm1d(mlp_layers[i + 1], eps=1e-05, momentum=0.1,
                                                       affine=True, track_running_stats=True) for i in
                                  range(len(mlp_layers) - 1)])

        # 最后一层
        self.fc2 = nn.Linear(mlp_layers[-1], 67)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)
        self.sigmoid = nn.Sigmoid()

    def forward(self, g, h):

        h = self.gcn1(g, h)
        h = self.gcn1_bn(h)
        h = self.act(self.dropout(h))

        for gcn2, bn2 in zip(self.gcn2, self.gcn2_bn):
            h = gcn2(g, h)
            h = bn2(h)
            h = self.act(self.dropout(h))

        # readout
        g.ndata['h'] = h
        x , atom_weights = self.readout(g, h, get_node_weight=True)

        x = self.bn1(self.fc1(x))
        x = self.dropout(x)
        x = self.act(x)
        for (i, linear), bn in zip(enumerate(self.linears), self.bns):
            x = linear(x)
            x = bn(x)
            x = self.dropout(x)
            x = self.act(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x, atom_weights



print('try')