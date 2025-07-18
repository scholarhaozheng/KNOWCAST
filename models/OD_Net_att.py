from torch_geometric import nn as gnn
from torch import nn
from torch.nn import functional as F
from torch.nn import init, Parameter
import torch
import random
import math
import sys
import os
import copy
import numpy as np
# from lib.utils_unfinished import softmax
from torch.nn import functional as F
sys.path.insert(0, os.path.abspath('..'))
from models.GATRUCell import GATRUCell
from models.GGRUCell import GGRUCell
from torch.nn import MultiheadAttention

class ODNet_att(torch.nn.Module):

    def __init__(self, cfg, logger):
        super(ODNet_att, self).__init__()
        self.logger = logger
        self.cfg = cfg

        self.trip_distribution_included = cfg['domain_knowledge_types_included']['trip_distribution']
        self.depart_freq_included = cfg['domain_knowledge_types_included']['depart_freq']
        self.traffic_assignment_included = cfg['domain_knowledge_types_included']['traffic_assignment']

        self.additional_section_feature_dim = cfg['model']['additional_section_feature_dim']
        self.additional_frequency_feature_dim = cfg['model']['additional_frequency_feature_dim']
        self.additional_distribution_dim = cfg['model']['additional_distribution_dim']

        self.history_distribution = cfg['domain_knowledge_loaded']['history_distribution']

        self.num_nodes = cfg['model']['num_nodes']
        self.num_output_dim = cfg['model']['output_dim']
        self.num_units = cfg['model']['rnn_units']
        self.num_decoder_input_dim = cfg['model']['input_dim']

        self.num_finished_input_dim = cfg['model']['input_dim']
        self.num_unfinished_input_dim = cfg['model']['input_dim']

        self.Using_GAT_or_RGCN = cfg['domain_knowledge']['Using_GAT_or_RGCN']

        if self.trip_distribution_included:
            self.num_finished_input_dim += self.additional_distribution_dim
            self.num_unfinished_input_dim += self.additional_distribution_dim
        if self.depart_freq_included:
            self.num_finished_input_dim += self.additional_frequency_feature_dim
            self.num_unfinished_input_dim += self.additional_frequency_feature_dim
        if self.traffic_assignment_included:
            self.num_finished_input_dim += self.additional_section_feature_dim
            self.num_unfinished_input_dim += self.additional_section_feature_dim

        self.num_rnn_layers = cfg['model']['num_rnn_layers']
        self.seq_len = cfg['model']['seq_len']
        self.horizon = cfg['model']['horizon']
        self.num_relations = cfg['model'].get('num_relations', 1)
        self.K = cfg['model'].get('K', 2)
        self.num_bases = cfg['model'].get('num_bases', 1)
        self.dropout_type = cfg['model'].get('dropout_type', None)
        self.dropout_prob = cfg['model'].get('dropout_prob', 0.0)
        self.batch_size = cfg['data']['batch_size']
        self.global_fusion = cfg['model'].get('global_fusion', False)

        self.gate_layer = nn.Conv1d(in_channels=self.num_units*4,
                                                 out_channels=self.num_units,
                                                 kernel_size=1)

        if self.Using_GAT_or_RGCN=="GCN":
            self.encoder_first_finished_cells = GGRUCell(self.num_finished_input_dim,
                                                         self.num_units,
                                                         self.dropout_type,
                                                         self.dropout_prob,
                                                         self.num_relations,
                                                         num_bases=self.num_bases,
                                                         K=self.K,
                                                         num_nodes=self.num_nodes,
                                                         global_fusion=self.global_fusion)
            self.encoder_first_unfinished_cells = GGRUCell(self.num_unfinished_input_dim,
                                                           self.num_units,
                                                           self.dropout_type,
                                                           self.dropout_prob,
                                                           self.num_relations,
                                                           num_bases=self.num_bases,
                                                           K=self.K,
                                                           num_nodes=self.num_nodes,
                                                           global_fusion=self.global_fusion)
            self.encoder_first_short_his_cells = GGRUCell(self.num_unfinished_input_dim,
                                                          self.num_units,
                                                          self.dropout_type,
                                                          self.dropout_prob,
                                                          self.num_relations,
                                                          num_bases=self.num_bases,
                                                          K=self.K,
                                                          num_nodes=self.num_nodes,
                                                          global_fusion=self.global_fusion)
            self.encoder_second_cells = nn.ModuleList([GGRUCell(self.num_units,
                                                                 self.num_units,
                                                                 self.dropout_type,
                                                                 self.dropout_prob,
                                                                 self.num_relations,
                                                                 num_bases=self.num_bases,
                                                                 K=self.K,
                                                                 num_nodes=self.num_nodes,
                                                                 global_fusion=self.global_fusion)
                                                       for _ in range(self.num_rnn_layers - 1)])
            self.decoder_first_cells = GGRUCell(self.num_decoder_input_dim,
                                                 self.num_units,
                                                 self.dropout_type,
                                                 self.dropout_prob,
                                                 self.num_relations,
                                                 num_bases=self.num_bases,
                                                 K=self.K,
                                                 num_nodes=self.num_nodes,
                                                 global_fusion=self.global_fusion)
            self.decoder_second_cells = nn.ModuleList([GGRUCell(self.num_units,
                                                                 self.num_units,
                                                                 self.dropout_type,
                                                                 self.dropout_prob,
                                                                 self.num_relations,
                                                                 self.K,
                                                                 num_nodes=self.num_nodes,
                                                                 global_fusion=self.global_fusion)
                                                       for _ in range(self.num_rnn_layers - 1)])

        else:
            self.encoder_first_finished_cells = GATRUCell(self.num_finished_input_dim,
                                                          self.num_units,
                                                          self.dropout_type,
                                                          self.dropout_prob,
                                                          self.num_relations,
                                                          num_bases=self.num_bases,
                                                          K=self.K,
                                                          num_nodes=self.num_nodes,
                                                          global_fusion=self.global_fusion)
            self.encoder_first_unfinished_cells = GATRUCell(self.num_unfinished_input_dim,
                                                            self.num_units,
                                                            self.dropout_type,
                                                            self.dropout_prob,
                                                            self.num_relations,
                                                            num_bases=self.num_bases,
                                                            K=self.K,
                                                            num_nodes=self.num_nodes,
                                                            global_fusion=self.global_fusion)
            self.encoder_first_short_his_cells = GATRUCell(self.num_unfinished_input_dim,
                                                           self.num_units,
                                                           self.dropout_type,
                                                           self.dropout_prob,
                                                           self.num_relations,
                                                           num_bases=self.num_bases,
                                                           K=self.K,
                                                           num_nodes=self.num_nodes,
                                                           global_fusion=self.global_fusion)
            self.encoder_second_cells = nn.ModuleList([GATRUCell(self.num_units,
                                                                 self.num_units,
                                                                 self.dropout_type,
                                                                 self.dropout_prob,
                                                                 self.num_relations,
                                                                 num_bases=self.num_bases,
                                                                 K=self.K,
                                                                 num_nodes=self.num_nodes,
                                                                 global_fusion=self.global_fusion)
                                                       for _ in range(self.num_rnn_layers - 1)])
            self.decoder_first_cells = GATRUCell(self.num_decoder_input_dim,
                                                 self.num_units,
                                                 self.dropout_type,
                                                 self.dropout_prob,
                                                 self.num_relations,
                                                 num_bases=self.num_bases,
                                                 K=self.K,
                                                 num_nodes=self.num_nodes,
                                                 global_fusion=self.global_fusion)
            self.decoder_second_cells = nn.ModuleList([GATRUCell(self.num_units,
                                                                 self.num_units,
                                                                 self.dropout_type,
                                                                 self.dropout_prob,
                                                                 self.num_relations,
                                                                 self.K,
                                                                 num_nodes=self.num_nodes,
                                                                 global_fusion=self.global_fusion)
                                                       for _ in range(self.num_rnn_layers - 1)])

        self.unfinished_output_layer = nn.Conv1d(in_channels=self.num_units*2,
                                                 out_channels=self.num_units,
                                                 kernel_size=1)
        self.unfinished_hidden_layer = nn.Conv1d(in_channels=self.num_units*2,
                                                 out_channels=self.num_units*2,
                                                 kernel_size=1)
        self.unfinished_hidden_double_layer = nn.Conv1d(in_channels=self.num_units * 2,
                                                 out_channels=self.num_units * 4,
                                                 kernel_size=1)
        self.gate_linear = nn.Linear(2, 2)

        self.attention_layer = MultiheadAttention(embed_dim=self.num_units*4, num_heads=2, dropout=self.dropout_prob)

        self.output_type = cfg['model'].get('output_type', 'fc')
        if self.output_type == 'fc':
            self.output_layer = nn.Linear(self.num_units, self.num_output_dim)

    def positional_encoding(self, seq_len, d_model):
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe = torch.zeros(seq_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0).transpose(0, 1)

    def encoder_first_layer(self,
                            *args,
                            **kwargs):
        if len(args) == 6:
            batch, finished_hidden, long_his_hidden, short_his_hidden, edge_index, edge_attr = args
            return self._encoder_first_layer_v1(batch, finished_hidden, long_his_hidden, short_his_hidden, edge_index,
                                                edge_attr)
        elif len(args) == 4:
            batch, finished_hidden, edge_index, edge_attr = args
            return self._encoder_first_layer_v2(batch, finished_hidden, edge_index, edge_attr)
        else:
            raise ValueError("Invalid arguments for encoder_first_layer")

    def _encoder_first_layer_v1(self, batch, finished_hidden, long_his_hidden, short_his_hidden, edge_index,
                                edge_attr=None):
        finished_out, finished_hidden = self.encoder_first_finished_cells(inputs=batch.x_od,
                                                                          edge_index=edge_index,
                                                                          edge_attr=edge_attr,
                                                                          hidden=finished_hidden)

        enc_first_hidden = finished_hidden
        enc_first_out = finished_out
        long_his_out, long_his_hidden = self.encoder_first_unfinished_cells(inputs=batch.history,
                                                                            edge_index=edge_index,
                                                                            edge_attr=edge_attr,
                                                                            hidden=long_his_hidden)

        short_his_out, short_his_hidden = self.encoder_first_short_his_cells(inputs=batch.yesterday,
                                                                             edge_index=edge_index,
                                                                             edge_attr=edge_attr,
                                                                             hidden=short_his_hidden)

        hidden_fusion = torch.cat([short_his_hidden, long_his_hidden], dim=-1).view(-1, self.num_units * 2, self.num_nodes).cuda()
        unfinished_hidden = self.unfinished_hidden_double_layer(hidden_fusion).cuda()
        # long_his_hidden:size(num_nodes*2, num_units)
        # short_his_hidden:size(num_nodes*2, num_units)
        # unfinished_hidden:size(batch_size,num_units*2,num_nodes)
        pos_enc = self.positional_encoding(unfinished_hidden.size(0), self.num_units * 4).cuda()
        pos_enc_permute = pos_enc.permute(0, 2, 1)
        unfinished_hidden = unfinished_hidden + pos_enc_permute
        # attention_layer: (sequence_length, batch_size, embedding_dim)
        unfinished_hidden_permute = unfinished_hidden.permute(2,0,1)# (num_nodes, batch_size, embedding_dim)
        attn_output, attn_weights = self.attention_layer(unfinished_hidden_permute, unfinished_hidden_permute, unfinished_hidden_permute)
        attn_output=self.gate_layer(attn_output.permute(1, 2, 0))
        enc_first_out = finished_out #+ attn_output.permute(2, 1, 0).reshape(self.num_nodes * self.batch_size, self.num_units)
        enc_first_hidden = enc_first_hidden #+ attn_output.permute(2, 1, 0).reshape(self.num_nodes * self.batch_size, self.num_units)
        return enc_first_out, finished_hidden, long_his_hidden, short_his_hidden, enc_first_hidden

    def _encoder_first_layer_v2(self, batch, finished_hidden, edge_index, edge_attr=None):
        finished_out, finished_hidden = self.encoder_first_finished_cells(inputs=batch.x_od,
                                                                          edge_index=edge_index,
                                                                          edge_attr=edge_attr,
                                                                          hidden=finished_hidden)
        enc_first_hidden = finished_hidden
        enc_first_out = finished_out
        return enc_first_out, finished_hidden, enc_first_hidden

    def encoder_second_layer(self,
                             index,
                             first_out,
                             enc_second_hidden,
                             edge_index,
                             edge_attr):
        enc_second_out, enc_second_hidden = self.encoder_second_cells[index](inputs=first_out,
                                                                             hidden=enc_second_hidden,
                                                                             edge_index=edge_index,
                                                                             edge_attr=edge_attr)
        return enc_second_out, enc_second_hidden

    def decoder_first_layer(self,
                            decoder_input,
                            dec_first_hidden,
                            edge_index,
                            edge_attr=None):
        dec_first_out, dec_first_hidden = self.decoder_first_cells(inputs=decoder_input,
                                                                   hidden=dec_first_hidden,
                                                                   edge_index=edge_index,
                                                                   edge_attr=edge_attr)
        return dec_first_out, dec_first_hidden

    def decoder_second_layer(self,
                             index,
                             decoder_first_out,
                             dec_second_hidden,
                             edge_index,
                             edge_attr=None):
        dec_second_out, dec_second_hidden = self.decoder_second_cells[index](inputs=decoder_first_out,
                                                                             hidden=dec_second_hidden,
                                                                             edge_index=edge_index,
                                                                             edge_attr=edge_attr)
        return dec_second_out, dec_second_hidden


