"""
DeepCE3D model integrating 2D Neural Fingerprint and 3D Graph Transformer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .neural_fingerprint import NeuralFingerprint
from .drug_gene_attention import DrugGeneAttention
from .graph_transformer_3d import GraphTransformer3D, convert_smile_to_3d_feature
from .ltr_loss import point_wise_mse, list_wise_listnet, list_wise_listmle, pair_wise_ranknet, list_wise_rankcosine, list_wise_ndcg


class FeatureFusion(nn.Module):
    """Feature fusion module for combining 2D and 3D molecular features"""
    
    def __init__(self, feature_dim, fusion_type='concat', dropout=0.1):
        """
        Args:
            feature_dim: dimension of features to fuse
            fusion_type: one of 'concat', 'add', 'gated', 'attention'
            dropout: dropout rate
        """
        super(FeatureFusion, self).__init__()
        self.fusion_type = fusion_type
        self.feature_dim = feature_dim
        
        if fusion_type == 'concat':
            # Output dimension is 2 * feature_dim
            pass
        elif fusion_type == 'add':
            # Output dimension is feature_dim
            pass
        elif fusion_type == 'gated':
            # Gated fusion with learnable gates
            self.gate = nn.Sequential(
                nn.Linear(feature_dim * 2, feature_dim),
                nn.Sigmoid()
            )
        elif fusion_type == 'attention':
            # Attention-based fusion
            self.attention = nn.Sequential(
                nn.Linear(feature_dim * 2, feature_dim),
                nn.Tanh(),
                nn.Linear(feature_dim, 2),
                nn.Softmax(dim=-1)
            )
        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")
            
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, feat_2d, feat_3d):
        """
        Args:
            feat_2d: 2D features [batch, num_atoms, feature_dim]
            feat_3d: 3D features [batch, num_atoms, feature_dim]
        Returns:
            fused features
        """
        if self.fusion_type == 'concat':
            # Simple concatenation
            fused = torch.cat([feat_2d, feat_3d], dim=-1)
            return self.dropout(fused)
            
        elif self.fusion_type == 'add':
            # Element-wise addition
            fused = feat_2d + feat_3d
            return self.dropout(fused)
            
        elif self.fusion_type == 'gated':
            # Gated fusion
            concat = torch.cat([feat_2d, feat_3d], dim=-1)
            gate = self.gate(concat)
            fused = gate * feat_2d + (1 - gate) * feat_3d
            return self.dropout(fused)
            
        elif self.fusion_type == 'attention':
            # Attention-based fusion
            concat = torch.cat([feat_2d, feat_3d], dim=-1)
            weights = self.attention(concat)  # [batch, num_atoms, 2]
            feat_2d_weighted = weights[:, :, 0:1] * feat_2d
            feat_3d_weighted = weights[:, :, 1:2] * feat_3d
            fused = feat_2d_weighted + feat_3d_weighted
            return self.dropout(fused)
    
    def get_output_dim(self):
        """Returns the output dimension after fusion"""
        if self.fusion_type == 'concat':
            return self.feature_dim * 2
        else:
            return self.feature_dim


class DeepCE3D(nn.Module):
    """Enhanced DeepCE with 3D Graph Transformer"""
    
    def __init__(self, drug_input_dim, drug_emb_dim, conv_size, degree, gene_input_dim, gene_emb_dim, num_gene,
                 hid_dim, dropout, loss_type, device, initializer=None, pert_type_input_dim=None,
                 cell_id_input_dim=None, pert_idose_input_dim=None,
                 pert_type_emb_dim=None, cell_id_emb_dim=None, pert_idose_emb_dim=None, use_pert_type=False,
                 use_cell_id=False, use_pert_idose=False,
                 # 3D-specific parameters
                 use_3d=True, fusion_type='concat', 
                 graph_3d_dim=128, graph_3d_n_heads=4, graph_3d_n_layers=2, graph_3d_d_ff=512):
        """
        Args:
            use_3d: whether to use 3D features
            fusion_type: type of feature fusion ('concat', 'add', 'gated', 'attention')
            graph_3d_dim: hidden dimension for 3D graph transformer
            graph_3d_n_heads: number of attention heads in 3D transformer
            graph_3d_n_layers: number of layers in 3D transformer
            graph_3d_d_ff: feedforward dimension in 3D transformer
        """
        super(DeepCE3D, self).__init__()
        assert drug_emb_dim == gene_emb_dim, 'Embedding size mismatch'
        self.use_pert_type = use_pert_type
        self.use_cell_id = use_cell_id
        self.use_pert_idose = use_pert_idose
        self.use_3d = use_3d
        self.drug_emb_dim = drug_emb_dim
        self.gene_emb_dim = gene_emb_dim

        # 2D Neural Fingerprint
        self.drug_fp = NeuralFingerprint(drug_input_dim['atom'], drug_input_dim['bond'], conv_size, drug_emb_dim,
                                         degree, device)
        
        # 3D Graph Transformer (optional)
        if self.use_3d:
            self.graph_3d = GraphTransformer3D(
                atom_input_dim=drug_input_dim['atom'],
                d_model=graph_3d_dim,
                n_heads=graph_3d_n_heads,
                n_layers=graph_3d_n_layers,
                d_ff=graph_3d_d_ff,
                output_dim=drug_emb_dim,
                dropout=dropout,
                device=device
            )
            # Feature fusion module
            self.fusion = FeatureFusion(drug_emb_dim, fusion_type, dropout)
            fused_dim = self.fusion.get_output_dim()
            self.fusion_proj = nn.Linear(fused_dim, drug_emb_dim)
            self.coords_cache = {}  # Cache for 3D coordinates
        
        self.gene_embed = nn.Linear(gene_input_dim, gene_emb_dim)

        self.drug_gene_attn = DrugGeneAttention(gene_emb_dim, gene_emb_dim, n_layers=2, n_heads=4, pf_dim=512,
                                                dropout=dropout, device=device)

        self.linear_dim = self.drug_emb_dim + self.gene_emb_dim

        if self.use_pert_type:
            self.pert_type_embed = nn.Linear(pert_type_input_dim, pert_type_emb_dim)
            self.linear_dim += pert_type_emb_dim
        if self.use_cell_id:
            self.cell_id_embed = nn.Linear(cell_id_input_dim, cell_id_emb_dim)
            self.linear_dim += cell_id_emb_dim
        if self.use_pert_idose:
            self.pert_idose_embed = nn.Linear(pert_idose_input_dim, pert_idose_emb_dim)
            self.linear_dim += pert_idose_emb_dim
        self.linear_1 = nn.Linear(self.linear_dim, hid_dim)
        self.linear_2 = nn.Linear(hid_dim, 1)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.num_gene = num_gene
        self.loss_type = loss_type
        self.initializer = initializer
        self.device = device
        self.init_weights()

    def init_weights(self):
        if self.initializer is None:
            return
        for name, parameter in self.named_parameters():
            if 'drug_gene_attn' not in name and 'graph_3d' not in name:
                if parameter.dim() == 1:
                    nn.init.constant_(parameter, 0.)
                else:
                    self.initializer(parameter)

    def forward(self, input_drug, input_gene, mask, input_pert_type, input_cell_id, input_pert_idose, smiles_batch=None):
        """
        Args:
            input_drug: dict with 'molecules', 'atom', 'bond'
            input_gene: gene features
            mask: attention mask
            input_pert_type, input_cell_id, input_pert_idose: additional features
            smiles_batch: list of SMILES strings for 3D coordinate generation (required if use_3d=True)
        """
        num_batch = input_drug['molecules'].batch_size
        
        # Extract 2D features
        drug_atom_embed_2d = self.drug_fp(input_drug)
        # drug_atom_embed_2d = [batch * num_node * drug_emb_dim]
        
        # Extract 3D features and fuse
        if self.use_3d:
            if smiles_batch is None:
                raise ValueError("smiles_batch is required when use_3d=True")
            
            # Generate 3D coordinates
            coords_3d, coords_mask = convert_smile_to_3d_feature(
                smiles_batch, drug_atom_embed_2d, self.device, cache=self.coords_cache
            )
            
            # Pad atom features to match 3D coords shape
            batch_size, max_atoms_3d, _ = coords_3d.shape
            _, max_atoms_2d, feat_dim = drug_atom_embed_2d.shape
            
            if max_atoms_3d > max_atoms_2d:
                # Pad 2D features
                padding = torch.zeros(batch_size, max_atoms_3d - max_atoms_2d, feat_dim).to(self.device).double()
                drug_atom_embed_2d_padded = torch.cat([drug_atom_embed_2d, padding], dim=1)
                atom_features_for_3d = input_drug['atom']
                # Pad atom input features as well
                atom_feat_dim = atom_features_for_3d.shape[-1]
                num_atoms_total = atom_features_for_3d.shape[0]
                # Reshape atom features to batch format
                batch_idx = input_drug['molecules'].get_neighbor_idx_by_batch('atom')
                atom_features_batch = torch.zeros(batch_size, max_atoms_3d, atom_feat_dim).to(self.device).double()
                for idx, atom_idx in enumerate(batch_idx):
                    atom_features_batch[idx, :len(atom_idx)] = atom_features_for_3d[atom_idx]
            elif max_atoms_3d < max_atoms_2d:
                # Truncate 2D features
                drug_atom_embed_2d_padded = drug_atom_embed_2d[:, :max_atoms_3d, :]
                batch_idx = input_drug['molecules'].get_neighbor_idx_by_batch('atom')
                atom_feat_dim = input_drug['atom'].shape[-1]
                atom_features_batch = torch.zeros(batch_size, max_atoms_3d, atom_feat_dim).to(self.device).double()
                for idx, atom_idx in enumerate(batch_idx):
                    num_atoms = min(len(atom_idx), max_atoms_3d)
                    atom_features_batch[idx, :num_atoms] = input_drug['atom'][atom_idx[:num_atoms]]
            else:
                drug_atom_embed_2d_padded = drug_atom_embed_2d
                batch_idx = input_drug['molecules'].get_neighbor_idx_by_batch('atom')
                atom_feat_dim = input_drug['atom'].shape[-1]
                atom_features_batch = torch.zeros(batch_size, max_atoms_3d, atom_feat_dim).to(self.device).double()
                for idx, atom_idx in enumerate(batch_idx):
                    atom_features_batch[idx, :len(atom_idx)] = input_drug['atom'][atom_idx]
            
            # Extract 3D features
            drug_atom_embed_3d = self.graph_3d(atom_features_batch, coords_3d, coords_mask)
            # drug_atom_embed_3d = [batch * num_atoms * drug_emb_dim]
            
            # Fuse 2D and 3D features
            fused_features = self.fusion(drug_atom_embed_2d_padded, drug_atom_embed_3d)
            drug_atom_embed = self.fusion_proj(fused_features)
            # drug_atom_embed = [batch * num_atoms * drug_emb_dim]
        else:
            drug_atom_embed = drug_atom_embed_2d
        
        # Continue with standard DeepCE pipeline
        drug_embed = torch.sum(drug_atom_embed, dim=1)
        # drug_embed = [batch * drug_emb_dim]
        drug_embed = drug_embed.unsqueeze(1)
        # drug_embed = [batch * 1 * drug_emb_dim]
        drug_embed = drug_embed.repeat(1, self.num_gene, 1)
        # drug_embed = [batch * num_gene * drug_emb_dim]
        gene_embed = self.gene_embed(input_gene)
        # gene_embed = [num_gene * gene_emb_dim]
        gene_embed = gene_embed.unsqueeze(0)
        # gene_embed = [1 * num_gene * gene_emb_dim]
        gene_embed = gene_embed.repeat(num_batch, 1, 1)
        # gene_embed = [batch * num_gene * gene_emb_dim]
        drug_gene_embed, _ = self.drug_gene_attn(gene_embed, drug_atom_embed, None, mask)
        # drug_gene_embed = [batch * num_gene * gene_emb_dim]
        drug_gene_embed = torch.cat((drug_gene_embed, drug_embed), dim=2)
        # drug_gene_embed = [batch * num_gene * (drug_emb_dim + gene_emb_dim)]
        if self.use_pert_type:
            pert_type_embed = self.pert_type_embed(input_pert_type)
            # pert_type_embed = [batch * pert_type_emb_dim]
            pert_type_embed = pert_type_embed.unsqueeze(1)
            # pert_type_embed = [batch * 1 * pert_type_emb_dim]
            pert_type_embed = pert_type_embed.repeat(1, self.num_gene, 1)
            # pert_type_embed = [batch * num_gene * pert_type_emb_dim]
            drug_gene_embed = torch.cat((drug_gene_embed, pert_type_embed), dim=2)
        if self.use_cell_id:
            cell_id_embed = self.cell_id_embed(input_cell_id)
            # cell_id_embed = [batch * cell_id_emb_dim]
            cell_id_embed = cell_id_embed.unsqueeze(1)
            # cell_id_embed = [batch * 1 * cell_id_emb_dim]
            cell_id_embed = cell_id_embed.repeat(1, self.num_gene, 1)
            # cell_id_embed = [batch * num_gene * cell_id_emb_dim]
            drug_gene_embed = torch.cat((drug_gene_embed, cell_id_embed), dim=2)
        if self.use_pert_idose:
            pert_idose_embed = self.pert_idose_embed(input_pert_idose)
            # pert_idose_embed = [batch * pert_idose_emb_dim]
            pert_idose_embed = pert_idose_embed.unsqueeze(1)
            # pert_idose_embed = [batch * 1 * pert_idose_emb_dim]
            pert_idose_embed = pert_idose_embed.repeat(1, self.num_gene, 1)
            # pert_idose_embed = [batch * num_gene * pert_idose_emb_dim]
            drug_gene_embed = torch.cat((drug_gene_embed, pert_idose_embed), dim=2)
        # drug_gene_embed = [batch * num_gene * (drug_embed + gene_embed + pert_type_embed + cell_id_embed + pert_idose_embed)]
        drug_gene_embed = self.relu(drug_gene_embed)
        # drug_gene_embed = [batch * num_gene * (drug_embed + gene_embed + pert_type_embed + cell_id_embed + pert_idose_embed)]
        out = self.linear_1(drug_gene_embed)
        # out = [batch * num_gene * hid_dim]
        out = self.relu(out)
        # out = [batch * num_gene * hid_dim]
        out = self.linear_2(out)
        # out = [batch * num_gene * 1]
        out = out.squeeze(2)
        # out = [batch * num_gene]
        return out

    def loss(self, label, predict):
        if self.loss_type == 'point_wise_mse':
            loss = point_wise_mse(label, predict)
        elif self.loss_type == 'pair_wise_ranknet':
            loss = pair_wise_ranknet(label, predict, self.device)
        elif self.loss_type == 'list_wise_listnet':
            loss = list_wise_listnet(label, predict)
        elif self.loss_type == 'list_wise_listmle':
            loss = list_wise_listmle(label, predict, self.device)
        elif self.loss_type == 'list_wise_rankcosine':
            loss = list_wise_rankcosine(label, predict)
        elif self.loss_type == 'list_wise_ndcg':
            loss = list_wise_ndcg(label, predict)
        else:
            raise ValueError('Unknown loss: %s' % self.loss_type)
        return loss
