"""
3D Graph Transformer for molecular 3D feature extraction
This module uses Graph Transformer to extract 3D molecular features from conformer coordinates
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np


class PositionalEncoding3D(nn.Module):
    """3D positional encoding based on inter-atomic distances"""
    
    def __init__(self, d_model, max_atoms=100, device='cpu'):
        super(PositionalEncoding3D, self).__init__()
        self.d_model = d_model
        self.max_atoms = max_atoms
        self.device = device
        self.distance_embed = nn.Sequential(
            nn.Linear(1, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, d_model)
        )
        
    def forward(self, coords):
        batch_size, num_atoms, _ = coords.shape
        diff = coords.unsqueeze(2) - coords.unsqueeze(1)
        distances = torch.sqrt((diff ** 2).sum(-1) + 1e-8)
        avg_distances = distances.mean(dim=-1, keepdim=True)
        pos_encoding = self.distance_embed(avg_distances)
        return pos_encoding


class GraphTransformerLayer3D(nn.Module):
    """Graph Transformer layer with 3D geometry awareness"""
    
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1, device='cpu'):
        super(GraphTransformerLayer3D, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.device = device
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.edge_bias = nn.Sequential(
            nn.Linear(1, n_heads),
            nn.ReLU()
        )
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, distances, mask=None):
        batch_size, num_atoms, _ = x.shape
        
        Q = self.W_q(x).view(batch_size, num_atoms, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, num_atoms, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, num_atoms, self.n_heads, self.d_k).transpose(1, 2)
        
        attn = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        dist_bias = self.edge_bias(distances.unsqueeze(-1))
        dist_bias = dist_bias.permute(0, 3, 1, 2)
        attn = attn + dist_bias
        
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)
            attn = attn.masked_fill(mask == 0, float('-inf'))
        
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(batch_size, num_atoms, self.d_model)
        out = self.W_o(out)
        
        x = self.norm1(x + self.dropout(out))
        out = self.ffn(x)
        x = self.norm2(x + out)
        
        return x


class GraphTransformer3D(nn.Module):
    """Graph Transformer for 3D molecular feature extraction"""
    
    def __init__(self, atom_input_dim, d_model, n_heads, n_layers, d_ff, 
                 output_dim, dropout=0.1, device='cpu'):
        super(GraphTransformer3D, self).__init__()
        self.d_model = d_model
        self.device = device
        
        self.atom_embed = nn.Linear(atom_input_dim, d_model)
        self.pos_encoding = PositionalEncoding3D(d_model, device=device)
        
        self.layers = nn.ModuleList([
            GraphTransformerLayer3D(d_model, n_heads, d_ff, dropout, device)
            for _ in range(n_layers)
        ])
        
        self.output_proj = nn.Linear(d_model, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, atom_features, coords_3d, mask=None):
        x = self.atom_embed(atom_features)
        pos_enc = self.pos_encoding(coords_3d)
        x = x + pos_enc
        x = self.dropout(x)
        
        diff = coords_3d.unsqueeze(2) - coords_3d.unsqueeze(1)
        distances = torch.sqrt((diff ** 2).sum(-1) + 1e-8)
        
        for layer in self.layers:
            x = layer(x, distances, mask)
        
        out = self.output_proj(x)
        return out


def generate_3d_coords(smiles, num_conformers=1, random_seed=42):
    """Generate 3D coordinates from SMILES using RDKit"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
            
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, randomSeed=random_seed)
        
        try:
            AllChem.MMFFOptimizeMolecule(mol)
        except:
            pass
            
        conf = mol.GetConformer()
        num_atoms = mol.GetNumAtoms()
        coords = np.zeros((num_atoms, 3))
        
        for i in range(num_atoms):
            pos = conf.GetAtomPosition(i)
            coords[i] = [pos.x, pos.y, pos.z]
            
        mol_no_h = Chem.RemoveHs(mol)
        num_heavy_atoms = mol_no_h.GetNumAtoms()
        coords = coords[:num_heavy_atoms]
        
        return coords
        
    except Exception as e:
        print(f"Error generating 3D coords for {smiles}: {e}")
        return None


def convert_smile_to_3d_feature(smiles_batch, atom_features_2d, device, cache=None):
    """Convert SMILES batch to 3D features"""
    batch_size = len(smiles_batch)
    coords_list = []
    
    for smiles in smiles_batch:
        if cache is not None and smiles in cache:
            coords = cache[smiles]
        else:
            coords = generate_3d_coords(smiles)
            if cache is not None:
                cache[smiles] = coords
                
        if coords is None:
            coords = np.zeros((1, 3))
            
        coords_list.append(coords)
    
    max_atoms = max(len(c) for c in coords_list)
    coords_padded = np.zeros((batch_size, max_atoms, 3))
    mask = np.zeros((batch_size, max_atoms))
    
    for i, coords in enumerate(coords_list):
        num_atoms = len(coords)
        coords_padded[i, :num_atoms] = coords
        mask[i, :num_atoms] = 1
    
    coords_3d = torch.FloatTensor(coords_padded).to(device).double()
    coords_mask = torch.FloatTensor(mask).to(device).double()
    
    return coords_3d, coords_mask
