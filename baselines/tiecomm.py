import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GATConv,GCNConv
import networkx as nx
import argparse
from modules.graph import measure_strength
from torch_geometric.data import Data


class GATLayerWithWeights(nn.Module):

    def __init__(self, in_features, out_features, dropout=0.1):
        super(GATLayerWithWeights, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout

        self.linear = nn.Linear(in_features, out_features, bias=False)

        self.att_src = nn.Parameter(torch.Tensor(1, out_features))
        self.att_dst = nn.Parameter(torch.Tensor(1, out_features))

        self.leaky_relu = nn.LeakyReLU(0.2)
        self.softmax = nn.Softmax(dim=0)
        self.dropout_layer = nn.Dropout(dropout)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.xavier_uniform_(self.att_src)
        nn.init.xavier_uniform_(self.att_dst)

    def forward(self, x, edge_index, return_attention_weights=True):
        """
        para:
            x: [n_nodes, in_features]
            edge_index: [2, n_edges]
        return:
            out: [n_nodes, out_features]
            attention_weights: [n_nodes, n_nodes] (稀疏存储为字典或矩阵)
        """
        h = self.linear(x)

        src, dst = edge_index
        edge_scores = (self.att_src * h[src] + self.att_dst * h[dst]).sum(dim=1)
        edge_scores = self.leaky_relu(edge_scores)

        n_nodes = x.size(0)
        attention_matrix = torch.zeros(n_nodes, n_nodes, device=x.device)

        unique_dst = torch.unique(dst)
        for node in unique_dst:
            mask = (dst == node)
            if mask.sum() > 0:
                incoming_src = src[mask]
                scores = edge_scores[mask]
                normalized_scores = self.softmax(scores)
                if self.training:
                    normalized_scores = self.dropout_layer(normalized_scores)
                attention_matrix[incoming_src, node] = normalized_scores

        # Aggregate neighbor information
        out = torch.zeros_like(h)
        for node in range(n_nodes):
            # Find the neighbors (source nodes) of this node
            neighbors = src[dst == node]
            if len(neighbors) > 0:
                weights = attention_matrix[neighbors, node].unsqueeze(1)  # [n_neighbors, 1]
                neighbor_features = h[neighbors]  # [n_neighbors, out_features]
                aggregated = torch.sum(weights * neighbor_features, dim=0)  # [out_features]
                out[node] = aggregated
            else:
                out[node] = h[node]
        if return_attention_weights:
            return out, attention_matrix
        else:
            return out

class TieCommAgent(nn.Module):

    def __init__(self, agent_config):
        super(TieCommAgent, self).__init__()

        self.args = argparse.Namespace(**agent_config)
        self.seed = self.args.seed

        self.n_agents = self.args.n_agents
        self.hid_size = self.args.hid_size

        self.core_selection = getattr(self.args, 'core_selection_method', 'original')

        self.agent = AgentAC(self.args)
        self.god = GodAC(self.args)

        if hasattr(self.args, 'random_prob'):
            self.random_prob = self.args.random_prob

        self.block = self.args.block

        self.hybrid_weight = getattr(self.args, 'hybrid_weight', 0.7)  #

        self.enhance_key_node_intra_obs = getattr(self.args, 'enhance_key_node_intra_obs', True)
        self.key_node_intra_enhancement = getattr(self.args, 'key_node_intra_enhancement', 2)


    def random_set(self):
        G = nx.binomial_graph(self.n_agents, self.random_prob, seed=self.seed , directed=False)
        sets = self.graph_partition(G, 0.5)
        return G, sets

    def graph_partition(self, G, god_action):

        #min_value = 0.5
        min_max_set = set()
        for e in G.edges():
            strength = measure_strength(G, e[0], e[1])
            G.add_edge(e[0], e[1], weight = round(strength,2))
            min_max_set.add(strength)


        min_max_list = list(min_max_set)
        if min_max_list:
            thershold = np.percentile(np.array(min_max_list), (int(god_action[0])) * 10)
        else:
            thershold = 0.0

        # thershold = 2

        g = nx.Graph()
        g.add_nodes_from(G.nodes(data=False), node_strength =0.0)

        # 初始化节点强度
        for node in g.nodes():
            g.nodes[node]['node_strength'] = 0.0

        for e in G.edges():
            strength = G.get_edge_data(e[0], e[1])['weight']
            if strength >= thershold:
                g.nodes[e[0]]['node_strength'] += strength
                g.nodes[e[1]]['node_strength'] += strength
                g.add_edge(e[0], e[1])
            # print(strength)
            # raise ValueError('strength > thershold')

        attr_dict = nx.get_node_attributes(g, 'node_strength')
        sets = []
        core_node = []

        original_cores = []
        node_strengths = []
        for c in nx.connected_components(g):
            list_c = list(c)
            sets.append(list_c)
            list_c_attr = [attr_dict[i] for i in list_c]

            team_strengths = [attr_dict[i] for i in list_c]
            node_strengths.append(team_strengths)

            original_core = list_c[list_c_attr.index(max(list_c_attr))]
            original_cores.append(original_core)

        extra_info = {
            'original_cores': original_cores,
            'node_strengths': node_strengths,
            'attr_dict': attr_dict
        }

        return g, (original_cores, sets, extra_info)  # 返回原始核心节点

    def recompute_core_nodes_with_strategy(self, sets, local_obs, intra_obs, graph,
                                           original_cores, attention_weights=None,
                                           extra_info=None):
        new_core_nodes = []
        all_team_key_nodes = []
        all_key_scores_list = []

        node_strengths = extra_info.get('node_strengths', []) if extra_info else []
        attr_dict = extra_info.get('attr_dict', {}) if extra_info else {}

        if isinstance(attention_weights, tuple):
            attention_weights = attention_weights[0] if len(attention_weights) > 0 else None

        for team_idx, team_members in enumerate(sets):
            team_size = len(team_members)
            original_core = original_cores[team_idx]

            key_scores = self.compute_intra_key_scores(local_obs, intra_obs, graph, attention_weights)

            # Make sure there are team members
            if len(team_members) > 0:
                team_indices = torch.tensor(team_members, device=key_scores.device)
                if team_indices.numel() > 0 and max(team_indices) < key_scores.size(0):
                    team_key_scores = key_scores[team_indices]
                else:
                    team_key_scores = torch.zeros(len(team_members), device=key_scores.device)
            else:
                team_key_scores = torch.tensor([], device=key_scores.device)

            if self.core_selection == 'original':
                new_core = original_core
                key_nodes = []

            #
            elif self.core_selection == 'key_node_only':
                if len(team_members) > 0 and team_key_scores.numel() > 0:
                    best_idx = torch.argmax(team_key_scores)
                    new_core = team_members[best_idx]
                    #
                    key_nodes, _ = self.select_key_nodes(team_members, key_scores, team_size)
                else:
                    new_core = original_core
                    key_nodes = []

            #
            elif self.core_selection == 'hybrid':
                if len(team_members) > 0 and team_key_scores.numel() > 0 and team_idx < len(node_strengths):
                    # 1. 归一化关键节点分数
                    if team_key_scores.numel() > 1:
                        min_key = torch.min(team_key_scores)
                        max_key = torch.max(team_key_scores)
                        if max_key > min_key:
                            normalized_key_scores = (team_key_scores - min_key) / (max_key - min_key)
                        else:
                            normalized_key_scores = torch.ones_like(team_key_scores) * 0.5
                    else:
                        normalized_key_scores = torch.ones_like(team_key_scores)
                    #ensure consistency in units
                    team_original_strengths = torch.tensor(
                        [attr_dict.get(i, 0.0) for i in team_members],
                        dtype=torch.float32,
                        device=team_key_scores.device
                    )

                    if team_original_strengths.numel() > 1:
                        min_strength = torch.min(team_original_strengths)
                        max_strength = torch.max(team_original_strengths)
                        if max_strength > min_strength:
                            normalized_strengths = (team_original_strengths - min_strength) / (
                                    max_strength - min_strength)
                        else:
                            normalized_strengths = torch.ones_like(team_original_strengths) * 0.5
                    else:
                        normalized_strengths = torch.ones_like(team_original_strengths)

                    alpha = self.hybrid_weight
                    weighted_scores = alpha * normalized_key_scores + (1 - alpha) * normalized_strengths

                    best_idx = torch.argmax(weighted_scores)
                    new_core = team_members[best_idx]

                    key_nodes, _ = self.select_key_nodes(team_members, key_scores, team_size)
                else:
                    new_core = original_core
                    key_nodes = []

            else:
                raise ValueError(f"Unknown core_selection method: {self.core_selection}")

            new_core_nodes.append(new_core)
            all_team_key_nodes.append(key_nodes)
            all_key_scores_list.append(team_key_scores)

        return new_core_nodes, all_team_key_nodes, all_key_scores_list

    def communicate(self, local_obs, graph=None, node_set =None):

        # core_node, set = node_set
        core_node, sets, extra_info = node_set
        original_cores = extra_info.get('original_cores', core_node)

        local_obs_emb = self.agent.local_emb(local_obs)

        intra_obs, attention_weights = self.agent.intra_com(local_obs_emb, graph)

        new_core_nodes, team_key_nodes, team_key_scores = self.recompute_core_nodes_with_strategy(
            sets, local_obs_emb, intra_obs, graph, original_cores, attention_weights, extra_info
        )

        all_key_nodes = [node for team in team_key_nodes for node in team]
        enhanced_local_obs = local_obs_emb.clone()

        if len(all_key_nodes) > 0:
            enhancement_factor = getattr(self.args, 'key_node_enhancement', 2)
            # Only enhance the key nodes
            for node_idx in all_key_nodes:
                # Find the index of this node within its team
                for team_idx, team_members in enumerate(sets):
                    if node_idx in team_members:
                        rel_idx = team_members.index(node_idx)
                        # Adjust the enhancement degree using the scores of key nodes
                        key_score = team_key_scores[team_idx][rel_idx]
                        score_factor = torch.sigmoid(key_score).item()
                        actual_enhancement = 1.0 + (enhancement_factor - 1.0) * score_factor
                        enhanced_local_obs[node_idx] = enhanced_local_obs[node_idx] * actual_enhancement
                        break

        # 5. Use appropriate observations for the final internal communication within the team
        if self.enhance_key_node_intra_obs:
            # 使用增强后的观测进行团队内通信
            intra_obs_final, attention_weights_final = self.agent.intra_com(enhanced_local_obs, graph)
        else:
            intra_obs_final = intra_obs
            attention_weights_final = attention_weights

        inter_obs = torch.zeros_like(intra_obs_final)
        if len(sets) != 1:

            team_reprs = []

            for team_idx, team_members in enumerate(sets):
                core_node = new_core_nodes[team_idx]
                key_nodes = team_key_nodes[team_idx]

                if len(key_nodes) > 0 and self.core_selection in ['key_node_only', 'hybrid']:
                    key_reprs = intra_obs_final[key_nodes, :]

                    key_indices = [team_members.index(k) for k in key_nodes]
                    key_weights = torch.softmax(team_key_scores[team_idx][key_indices], dim=0)

                    team_repr = torch.sum(key_reprs * key_weights.unsqueeze(1), dim=0)
                else:
                    team_repr = intra_obs_final[core_node, :]

                team_reprs.append(team_repr)

            team_reprs_tensor = torch.stack(team_reprs, dim=0)
            group_obs = self.agent.inter_com(team_reprs_tensor)

            for idx, team_members in enumerate(sets):
                inter_obs[team_members, :] = group_obs[idx, :].repeat(len(team_members), 1)

        # 7. Construct the final representation
        if self.block == 'no':
            after_comm = torch.cat((enhanced_local_obs, inter_obs, intra_obs_final), dim=-1)
        elif self.block == 'inter':
            after_comm = torch.cat((enhanced_local_obs, intra_obs_final, torch.rand_like(inter_obs)), dim=-1)
        elif self.block == 'intra':
            after_comm = torch.cat((enhanced_local_obs, inter_obs, torch.rand_like(intra_obs_final)), dim=-1)
        else:
            raise ValueError('block must be one of no, inter, intra')

        # Return the updated information of the node set
        updated_extra_info = {
            'original_cores': original_cores,
            'key_nodes': team_key_nodes,
            'key_scores': team_key_scores,
            'core_selection': self.core_selection,
            'enhance_key_node_intra_obs': self.enhance_key_node_intra_obs,
            'attention_weights': attention_weights_final.detach().cpu().numpy() if attention_weights_final is not None else None
        }

        return after_comm, (new_core_nodes, sets, updated_extra_info)

    def compute_intra_key_scores(self, local_obs, intra_obs, graph, attention_weights=None):
        n_agents = local_obs.shape[0]
        device = local_obs.device

        base_scores = torch.zeros(n_agents, device=device)

        if attention_weights is not None:
            if isinstance(attention_weights, tuple):
                attention_weights = attention_weights[0] if len(attention_weights) > 0 else None

            if isinstance(attention_weights, torch.Tensor):

                attention_weights = F.softmax(attention_weights.view(-1), dim=0).view_as(attention_weights)

                # The diversity of attention distribution (using entropy)
                attention_entropy = -torch.sum(attention_weights * torch.log(attention_weights + 1e-8), dim=-1)

                # Encourage diversified distribution of attention
                diversity_score = attention_entropy / torch.log(torch.tensor(n_agents, dtype=torch.float))
                base_scores += 0.2 * diversity_score

                # Attentional Centrality Score (with maximum value limited)
                attention_in = attention_weights.sum(dim=0)
                attention_out = attention_weights.sum(dim=1)

                attention_in_norm = F.softmax(attention_in, dim=0)
                attention_out_norm = F.softmax(attention_out, dim=0)

                base_scores += 0.5 * attention_in_norm + 0.5 * attention_out_norm

        degrees = []
        for i in range(n_agents):
            if i in graph:
                degrees.append(graph.degree(i))
            else:
                degrees.append(0)
        degree_tensor = torch.tensor(degrees, device=device, dtype=torch.float)
        if degree_tensor.max() > 0:
            degree_score = degree_tensor / degree_tensor.max()
            base_scores += 0.1 * degree_score

        # 3. repres norm (normalization)
        norm_score = torch.norm(intra_obs, dim=1)
        if norm_score.max() > 0:
            norm_score_norm = norm_score / norm_score.max()
            base_scores += 0.2 * norm_score_norm

        # 4. The amount of local observation information (normalized)
        if local_obs.size(1) > 1:
            obs_variance = torch.var(local_obs, dim=1)
            if obs_variance.max() > 0:
                obs_variance_norm = obs_variance / obs_variance.max()
                base_scores += 0.1 * obs_variance_norm
        # 6. Obtain the final probability distribution using softmax.
        key_scores_prob = F.softmax(base_scores, dim=0)

        return key_scores_prob

    def select_key_nodes(self, team_members, key_scores, team_size, min_key_nodes=1):

        base_key_nodes = min_key_nodes
        scale_factor = getattr(self.args, 'key_node_scaling', 0.2)
        num_key_nodes = max(base_key_nodes,
                            int(base_key_nodes + (team_size - 5) * scale_factor))
        num_key_nodes = min(num_key_nodes, team_size)

        if num_key_nodes == 0:
            num_key_nodes = 1

        team_indices = torch.tensor(team_members, device=key_scores.device)

        if team_indices.numel() > 0:
            valid_indices = team_indices[team_indices < key_scores.size(0)]
            if valid_indices.numel() > 0:
                team_scores = key_scores[valid_indices]
            else:
                team_scores = torch.zeros(len(team_members), device=key_scores.device)
        else:
            team_scores = torch.zeros(len(team_members), device=key_scores.device)

        if team_scores.numel() > 0 and num_key_nodes > 0:

            if team_scores.sum() > 0:
                probs = team_scores / team_scores.sum()
            else:
                probs = torch.ones_like(team_scores) / len(team_scores)

            temperature = getattr(self.args, 'key_node_temperature', 1.0)
            if temperature != 1.0:
                probs = F.softmax(team_scores / temperature, dim=0)

            probs = probs / probs.sum()

            replace = getattr(self.args, 'key_node_replace', False)
            indices = torch.multinomial(probs, num_samples=min(num_key_nodes, probs.numel()),
                                        replacement=replace)
            # A certain probability will use top-k, and a certain probability will use sampling
            use_topk_prob = getattr(self.args, 'use_topk_prob', 0.7)
            if torch.rand(1).item() < use_topk_prob:
                # Use top-k (deterministic selection)
                _, topk_indices = torch.topk(team_scores, k=min(num_key_nodes, team_scores.numel()))
                key_nodes = [team_members[i] for i in topk_indices.cpu().numpy()]
                key_importance = torch.softmax(team_scores[topk_indices], dim=0)
            else:
                # Use sampling (random exploration)
                key_nodes = [team_members[i] for i in indices.cpu().numpy()]
                key_importance = torch.softmax(team_scores[indices], dim=0)
        else:
            key_nodes = []
            key_importance = torch.tensor([], device=key_scores.device)

        return key_nodes, key_importance


class GodAC(nn.Module):
    def __init__(self, args):
        super(GodAC, self).__init__()
        self.args = args
        self.n_agents = args.n_agents
        self.hid_size = args.hid_size
        self.threshold = self.args.threshold
        self.tanh = nn.Tanh()

        self.fc1_1 = nn.Linear(args.obs_shape * self.n_agents , self.hid_size * 1)
        self.fc1_2 = nn.Linear(self.n_agents**2 , self.hid_size)
        self.fc2 = nn.Linear(self.hid_size *2, self.hid_size)
        self.head = nn.Linear(self.hid_size, 10)
        self.value = nn.Linear(self.hid_size, 1)


    def forward(self, input, graph):

        adj_matrix = torch.tensor(nx.to_numpy_array(graph), dtype=torch.float).view(1, -1)
        h1 = self.tanh(self.fc1_1(input.view(1, -1)))
        h2 = self.tanh(self.fc1_2(adj_matrix))
        hid = torch.cat([h1,h2], dim=1)
        hid = self.tanh(self.fc2(hid))

        a = F.log_softmax(self.head(hid), dim=-1)
        v = self.value(hid)

        return a, v






class AgentAC(nn.Module):
    def __init__(self, args):
        super(AgentAC, self).__init__()

        self.args = args
        self.n_agents = args.n_agents
        self.hid_size = 32
        self.n_actions = self.args.n_actions
        self.tanh = nn.Tanh()

        self.emb_fc = nn.Linear(args.obs_shape, self.hid_size)

        # self.intra = GATConv(self.hid_size, self.hid_size, heads=1, add_self_loops =False, concat=False,dropout=0.0)
        # self.intra = GCNConv(self.hid_size, self.hid_size, add_self_loops= False)

        self.intra = GATLayerWithWeights(self.hid_size, self.hid_size, dropout=0.1)

        #encoder_layer = nn.TransformerEncoderLayer(d_model=self.hid_size, nhead=1, dim_feedforward=self.hid_size,
        #                                                batch_first=True)
        #self.inter = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.inter = nn.MultiheadAttention(self.hid_size, num_heads=1, batch_first=True)
        # self.affine2 = nn.Linear(self.hid_size * 3 + self.n_agents**2  , self.hid_size)
        self.affine2 = nn.Linear(self.hid_size * 3, self.hid_size)


        self.actor_head = nn.Linear(self.hid_size, self.n_actions)
        self.value_head = nn.Linear(self.hid_size, 1)

    def local_emb(self, input):
        local_obs = self.tanh(self.emb_fc(input))
        return local_obs

    def intra_com(self, x, graph):
        if x.size(1) != self.hid_size:
            print(f"Warning: intra_com input dimension mismatch. Expected: {self.hid_size}, Got: {x.size(1)}")
            x = self.local_emb(x)

        if list(graph.edges()) == []:
            h = torch.zeros(x.shape[0], self.hid_size, device=x.device)
            attention_weights = torch.zeros(x.shape[0], x.shape[0], device=x.device)
        else:
            edges = list(graph.edges())
            edge_index = torch.tensor(edges, dtype=torch.long, device=x.device)
            edge_index = edge_index.t().contiguous()  # [2, num_edges]
            result = self.intra(x, edge_index, return_attention_weights=True)

            if isinstance(result, tuple) and len(result) == 2:
                h, attention_weights = result
            else:
                print('intra_com error')

            h = self.tanh(h)

        return h, attention_weights

    def inter_com(self, input):
        x = input.unsqueeze(0)
        h, weights = self.inter(x, x, x)
        #h = self.inter(x)

        return h.squeeze(0)




    def forward(self, final_obs):
        h = self.tanh(self.affine2(final_obs))
        a = F.log_softmax(self.actor_head(h), dim=-1)
        v = self.value_head(h)

        return a, v





#
# class GodActor(nn.Module):
#     def __init__(self, args):
#         super(GodActor, self).__init__()
#         self.args = args
#         self.n_agents = args.n_agents
#         self.hid_size = self.hid_size
#         self.threshold = self.args.threshold
#         self.tanh = nn.Tanh()
#
#         self.fc1 = nn.Linear(args.obs_shape * self.n_agents + self.n_agents**2 , self.hid_size * 3)
#         self.fc2 = nn.Linear(self.hid_size * 3 , self.hid_size)
#         self.head = nn.Linear(self.hid_size, 10)
#
#     def forward(self, input, graph):
#
#         adj_matrix = torch.tensor(nx.to_numpy_array(graph), dtype=torch.float).view(1, -1)
#         hid = torch.cat([input.view(1,-1), adj_matrix], dim=1)
#         hid = self.tanh(self.fc1(hid))
#         hid = self.tanh(self.fc2(hid))
#         a = F.log_softmax(self.head(hid), dim=-1)
#
#         return a
#
#
#
# class GodCritic(nn.Module):
#     def __init__(self, args):
#         super(GodCritic, self).__init__()
#         self.args = args
#         self.n_agents = args.n_agents
#         self.hid_size = args.hid_size
#         self.threshold = self.args.threshold
#         self.tanh = nn.ReLU()
#
#         self.fc1 = nn.Linear(args.obs_shape * self.n_agents + self.n_agents ** 2, self.hid_size * 4)
#         self.fc2 = nn.Linear(self.hid_size * 4 , self.hid_size)
#         self.value = nn.Linear(self.hid_size, 1)
#
#     def forward(self, input, graph):
#
#         adj_matrix = torch.tensor(nx.to_numpy_array(graph), dtype=torch.float).view(1, -1)
#         hid = torch.cat([input.view(1,-1), adj_matrix], dim=1)
#         hid = self.tanh(self.fc1(hid))
#         hid = self.tanh(self.fc2(hid))
#         v = self.value(hid)
#
#         return v
#