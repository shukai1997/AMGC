import torch
from dgllife.utils import BaseAtomFeaturizer
from dgllife.utils import atom_type_one_hot,\
                 atom_degree_one_hot,\
                 atom_implicit_valence_one_hot,\
                 atom_formal_charge,\
                 atom_num_radical_electrons,\
                 atom_hybridization_one_hot,\
                 atom_is_aromatic,\
                 atom_total_num_H_one_hot,\
                 atom_chiral_tag_one_hot
from dgllife.utils import ConcatFeaturizer
from dgllife.utils import smiles_to_bigraph,mol_to_bigraph
from dgllife.utils import CanonicalBondFeaturizer
from dgl.data.utils import save_graphs
from rdkit.Chem import PandasTools
from functools import partial



class MyAtomFeaturizer(BaseAtomFeaturizer):
    def __init__(self, atom_data_filed='h'):
        super(MyAtomFeaturizer, self).__init__(
            featurizer_funcs={atom_data_filed: ConcatFeaturizer([partial(atom_type_one_hot, allowable_set=['C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Br', 'I', 'B', 'H', 'Si' , 'Se'], encode_unknown=True),
                                                                 partial(atom_degree_one_hot,
                                                                         allowable_set=list(range(6))),
                                                                 atom_formal_charge, atom_num_radical_electrons,
                                                                 partial(atom_hybridization_one_hot,
                                                                         encode_unknown=True),
                                                                 atom_is_aromatic,
                                                                 # A placeholder for aromatic information,
                                                                 atom_total_num_H_one_hot, atom_chiral_tag_one_hot])})



