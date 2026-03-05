import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import argparse

class TeamCommAgent(nn.Module):
    def __init__(self, agent_config):
        super(TeamCommAgent, self).__init__()

        self.args = argparse.Namespace(**agent_config)
        self.seed = self.args.seed

        self.n_agents = self.args.n_agents
        self.hid_size = self.args.hid_size

        self.agent = AgentAC(self.args)
        self.teaming = Teaming(self.args)


        self.block = self.args.block

        # === add-Pooling mode parameters ===
        self.pooling_mode = getattr(self.args, 'pooling_mode', 'original')  # original, weighted, key_only
        self.pooling_key_weight = getattr(self.args, 'pooling_key_weight', 1.2)  # Key node weight multiplier
        self.pooling_key_only_threshold = getattr(self.args, 'pooling_key_only_threshold', 0.5)  # Key node proportion threshold

        self.env_type = getattr(self.args, 'env_type', 'mpe')  # add env

        self.use_intra_key_nodes = getattr(self.args, 'use_intra_key_nodes', True)
        self.use_inter_key_teams = getattr(self.args, 'use_inter_key_teams', True)
        self.use_comm_weight_enhancement = getattr(self.args, 'use_comm_weight_enhancement', True)
        self.comm_weight_enhance_factor = getattr(self.args, 'comm_weight_enhance_factor', 0.1)

        self.use_obs_enhancement = getattr(self.args, 'use_obs_enhancement', True)
        self.obs_enhance_factor = getattr(self.args, 'obs_enhance_factor', 0.05)

        self.intra_struct_weight = getattr(self.args, 'intra_struct_weight', 0.6)
        self.key_node_threshold_factor = getattr(self.args, 'key_node_threshold_factor', 1.0)

        self.use_key_node_for_vib = getattr(self.args, 'use_key_node_for_vib', True)
        self.key_node_vib_weight = getattr(self.args, 'key_node_vib_weight', 0.6)


    def communicate(self, obs, sets, team_action_out=None, step=0):
        local_obs = self.agent.local_emb(obs)
        inter_obs_emb = self.agent.inter_emb(obs)
        intra_obs_emb = self.agent.intra_emb(obs)

        inter_obs = torch.zeros_like(inter_obs_emb)
        intra_obs = torch.zeros_like(intra_obs_emb)

        inter_mu = torch.zeros_like(inter_obs)
        inter_std = torch.zeros_like(inter_obs)
        intra_mu = torch.zeros_like(intra_obs)
        intra_std = torch.zeros_like(intra_obs)

        # Store key node information - Add the 'team_key_masks' key
        key_node_info = {
            'intra_key_nodes': [],
            'inter_key_teams': [],
            'intra_scores': [],
            'inter_scores': [],
            'adjusted_obs': obs.clone(),  # ori-obs
            'key_node_mask': torch.zeros(self.n_agents, device=obs.device),
            'team_ids': torch.zeros(self.n_agents, dtype=torch.long, device=obs.device),
            'team_key_masks': {}  # key_mask
        }

        n_groups = len(sets)

        # === Build the team ID mapping ===
        for team_idx, team_set in enumerate(sets):
            for agent_idx in team_set:
                key_node_info['team_ids'][agent_idx] = team_idx

        # === Identification of internal communication channels and key nodes for all teams ===
        for team_idx, team_set in enumerate(sets):
            team_key_mask = torch.zeros(len(team_set), device=obs.device)

            if len(team_set) > 1:
                # Intra-team communication (original, unadjusted)
                intra_team_obs = intra_obs_emb[team_set, :]
                intra_team_output, intra_team_mu, intra_team_std, intra_attn_weights = self.agent.intra_com(
                    intra_team_obs)

                # only calculated when the key nodes within the team are enabled.
                if self.use_intra_key_nodes:
                    intra_key_scores = self.compute_intra_key_scores(
                        intra_team_obs,
                        intra_team_mu,
                        intra_attn_weights,
                        len(team_set),
                        n_groups
                    )

                    intra_key_nodes = self.identify_key_nodes(
                        intra_key_scores,
                        len(team_set),
                        n_groups,
                        node_type='intra'
                    )

                    # Update the key node masks of the current team
                    for node_idx in intra_key_nodes:
                        team_key_mask[node_idx] = 1.0

                    # adjust the IB of the key nodes themselves
                    if intra_key_nodes:
                        # avoid in-place operations
                        adjusted_intra_team_mu, adjusted_intra_team_std = self.adjust_key_node_bottleneck(
                            intra_team_mu,
                            intra_team_std,
                            intra_key_nodes,
                            key_scores=intra_key_scores
                        )
                        # Re-calculate the outputs of the key nodes
                        intra_team_output = self.agent.reparameterise(adjusted_intra_team_mu, adjusted_intra_team_std)

                        # Store the adjusted results
                        intra_mu[team_set, :] = adjusted_intra_team_mu
                        intra_std[team_set, :] = adjusted_intra_team_std
                        intra_obs[team_set, :] = intra_team_output
                    else:
                        # Store the original communication results
                        intra_obs[team_set, :] = intra_team_output
                        intra_mu[team_set, :] = intra_team_mu
                        intra_std[team_set, :] = intra_team_std

                    # Store global key node information
                    for node_idx in intra_key_nodes:
                        actual_agent_idx = team_set[node_idx]
                        key_node_info['intra_key_nodes'].append(actual_agent_idx)
                        key_node_info['intra_scores'].append(intra_key_scores[node_idx].item())
                else:
                    # Use the original results
                    intra_obs[team_set, :] = intra_team_output
                    intra_mu[team_set, :] = intra_team_mu
                    intra_std[team_set, :] = intra_team_std
            else:
                # Individual team
                intra_obs[team_set, :] = intra_obs_emb[team_set, :]
                intra_mu[team_set, :] = intra_obs_emb[team_set, :]
                intra_std[team_set, :] = torch.ones_like(intra_obs_emb[team_set, :]) * 0.1

            # store
            key_node_info['team_key_masks'][team_idx] = team_key_mask

        # === Pooling based on key node information ===
        global_set = []

        for team_idx, team_set in enumerate(sets):
            member_inter_obs = inter_obs_emb[team_set, :]

            # get key_mask
            team_key_mask = key_node_info['team_key_masks'][team_idx]

            if self.pooling_mode == 'original':

                team_pooling = self.agent.pooling(member_inter_obs)
            elif self.pooling_mode == 'weighted':

                team_pooling = self._weighted_pooling_with_key_nodes(
                    member_inter_obs, team_key_mask
                )
            elif self.pooling_mode == 'key_only':

                team_pooling = self._key_only_pooling(
                    member_inter_obs, team_key_mask
                )
            else:

                team_pooling = self.agent.pooling(member_inter_obs)

            global_set.append(team_pooling)

        # === Inter-team communication ===
        if len(global_set) > 1:
            inter_obs_input = torch.cat(global_set, dim=0)
            inter_obs_output, inter_mu_output, inter_std_output, inter_attn_weights = self.agent.inter_com(
                inter_obs_input)

            # store ori
            for idx, team_set in enumerate(sets):
                if len(team_set) > 1:
                    inter_obs[team_set, :] = inter_obs_output[idx, :].repeat(len(team_set), 1)
                    inter_mu[team_set, :] = inter_mu_output[idx, :].repeat(len(team_set), 1)
                    inter_std[team_set, :] = inter_std_output[idx, :].repeat(len(team_set), 1)
                else:
                    inter_obs[team_set, :] = inter_obs_output[idx, :]
                    inter_mu[team_set, :] = inter_mu_output[idx, :]
                    inter_std[team_set, :] = inter_std_output[idx, :]

        # === Adjust communication based on key node information (to prevent collapse) ===
        # Only adjust the internal communication within the team
        if self.use_comm_weight_enhancement and self.use_intra_key_nodes:
            adjusted_intra_obs = self.adjust_communication_safe(
                intra_obs,
                key_node_info,
                sets,
                adjustment_type='intra'
            )
        else:
            adjusted_intra_obs = intra_obs

        # make adjustments
        adjusted_obs = obs.clone()
        if self.use_obs_enhancement and key_node_info['intra_key_nodes']:
            for agent_idx in key_node_info['intra_key_nodes']:
                if agent_idx < obs.shape[0]:
                    #
                    adjusted_obs[agent_idx] = obs[agent_idx] * (1.0 + self.obs_enhance_factor)

        # store
        key_node_info['adjusted_obs'] = adjusted_obs
        key_node_info['original_obs'] = obs

        # a global key node mask
        key_node_mask = torch.zeros(self.n_agents, device=obs.device)
        for agent_idx in key_node_info['intra_key_nodes']:
            if agent_idx < self.n_agents:
                key_node_mask[agent_idx] = 1.0
        key_node_info['key_node_mask'] = key_node_mask

        # ensure team_ids is tensor
        if not isinstance(key_node_info['team_ids'], torch.Tensor):
            key_node_info['team_ids'] = torch.tensor(key_node_info['team_ids'], device=obs.device)

        adjusted_local_obs = self.agent.local_emb(key_node_info['adjusted_obs'])
        if self.block == 'no':
            # patch
            after_comm = torch.cat((adjusted_local_obs, inter_obs, adjusted_intra_obs), dim=-1)
        elif self.block == 'inter':
            after_comm = torch.cat((local_obs, intra_obs, torch.rand_like(inter_obs)), dim=-1)
        elif self.block == 'intra':
            after_comm = torch.cat((local_obs, inter_obs, torch.rand_like(intra_obs)), dim=-1)
        else:
            raise ValueError('block must be one of no, inter, intra')

        mu = torch.cat((intra_mu, inter_mu), dim=-1)
        std = torch.cat((intra_std, inter_std), dim=-1)

        return after_comm, mu, std, key_node_info

    def adjust_key_node_bottleneck(self, mu, std, key_indices, key_scores=None):
        """只调整关键节点自身的信息瓶颈"""
        if not key_indices or not self.use_key_node_for_vib:
            return mu, std

            # copy
        adjusted_mu = mu.detach().clone()
        adjusted_std = std.detach().clone()

        for idx in key_indices:
            if idx < adjusted_mu.shape[0]:
                # use key_node_vib_weight para
                vib_weight = self.key_node_vib_weight

                # based vib_weight
                std_factor = 1.0 + (1.0 - vib_weight) * 1.5
                mu_factor = 1.0 + (1.0 - vib_weight) * 0.5

                # if has key_scores，
                if key_scores is not None and idx < len(key_scores):
                    score = key_scores[idx]
                    # based key_scores
                    score_factor = score / (key_scores.mean() + 1e-8)
                    std_factor = std_factor * score_factor
                    mu_factor = mu_factor * score_factor

                # Ensure that no in-place operations are performed
                new_mu_val = adjusted_mu[idx] * mu_factor
                new_std_val = adjusted_std[idx] * std_factor

                # establish security
                adjusted_mu = torch.cat([
                    adjusted_mu[:idx],
                    new_mu_val.unsqueeze(0),
                    adjusted_mu[idx + 1:]
                ]) if idx < adjusted_mu.shape[0] - 1 else torch.cat([
                    adjusted_mu[:idx],
                    new_mu_val.unsqueeze(0)
                ])

                adjusted_std = torch.cat([
                    adjusted_std[:idx],
                    new_std_val.unsqueeze(0),
                    adjusted_std[idx + 1:]
                ]) if idx < adjusted_std.shape[0] - 1 else torch.cat([
                    adjusted_std[:idx],
                    new_std_val.unsqueeze(0)
                ])
        adjusted_std = torch.clamp(adjusted_std, 0.1, 3.0)

        return adjusted_mu, adjusted_std

    def compute_intra_key_scores(self, team_obs, attention_mu, attn_weights, team_size, n_groups):
        """Calculation of key node scores within the team utilizes attention_mu and takes into account the team size."""
        n_members = team_obs.shape[0]

        # 1. Information measurement
        attention_probs = F.softmax(attn_weights, dim=-1)
        entropy = -torch.sum(attention_probs * torch.log(attention_probs + 1e-8), dim=0)
        max_entropy = torch.log(torch.tensor(n_members, dtype=torch.float))
        info_metric = 1.0 - entropy / max_entropy

        # 2. Structural measurement
        in_attention = attn_weights.sum(dim=0)
        out_attention = attn_weights.sum(dim=1)

        # 3. Indicate importance
        if attention_mu.dim() == 3:
            mu_features = attention_mu.squeeze(0)
        else:
            mu_features = attention_mu

        mu_norms = torch.norm(mu_features, dim=1)
        representation_metric = mu_norms / (mu_norms.max() + 1e-8)

        # 4. Team size adjustment factor
        team_size_factor = min(1.0, team_size / 10.0)

        group_factor = 1.5 if n_groups == 1 else 1.0

        # 6. Comprehensive structural measurement, taking into account the team size
        structure_metric = (0.6 * (in_attention / (in_attention.max() + 1e-8)) +
                            0.3 * (out_attention / (out_attention.max() + 1e-8)) +
                            0.1 * representation_metric) * team_size_factor

        # 7. confuse
        info_metric_norm = F.softmax(info_metric, dim=0)
        structure_metric_norm = F.softmax(structure_metric, dim=0)

        log_info = torch.log(info_metric_norm + 1e-8)
        log_structure = torch.log(structure_metric_norm + 1e-8)

        # The number of teams affects the allocation of weights.
        if n_groups == 1:
            intra_weight = 0.8
        else:
            intra_weight = self.intra_struct_weight

        weighted_log = ((1 - intra_weight) * log_info + intra_weight * log_structure)

        key_scores = F.softmax(weighted_log, dim=0) * group_factor

        # 8. temper
        temperature = 1.2 if n_groups == 1 else 1.0  # 单一团队时更平滑
        key_scores = F.softmax(torch.log(key_scores + 1e-8) / temperature, dim=0)

        return key_scores

    def _weighted_pooling_with_key_nodes(self, member_inter_obs, team_key_mask):
        """
        Weighted Pooling: Higher weights for key nodes
        Use the key node mask of the current team
        """
        return self.agent.pooling(
            member_inter_obs,
            key_node_mask=team_key_mask,
            key_weight_factor=self.pooling_key_weight
        )

    def _key_only_pooling(self, member_inter_obs, team_key_mask):

        n_members = len(member_inter_obs)

        # Obtain the key node indices of the current team
        key_indices = torch.where(team_key_mask > 0.5)[0]

        if len(key_indices) == 0:
            # If there are no critical nodes, revert to weighted pooling
            return self._weighted_pooling_with_key_nodes(member_inter_obs, team_key_mask)

        # Check whether the proportion of key nodes has reached the threshold
        key_ratio = len(key_indices) / n_members
        if key_ratio < self.pooling_key_only_threshold:
            # There are too few key nodes, so we have to revert to weighted pooling.
            return self._weighted_pooling_with_key_nodes(member_inter_obs, team_key_mask)

        # Only using the representation of key nodes
        key_member_obs = member_inter_obs[key_indices]

        # Calculate the pooling of key nodes
        return self.agent.pooling(key_member_obs)

    def identify_key_nodes(self, scores, n_nodes, n_groups, node_type='intra'):

        if len(scores) == 0:
            return []

        mean_score = scores.mean()
        std_score = scores.std()

        #Dynamic adjustment of threshold
        if n_groups == 1:

            threshold_factor = getattr(self.args, 'key_node_threshold_factor', 1.5) * 1.2
            max_key_ratio = 0.1  # 最多10%的节点作为关键节点
        elif n_groups <= max(2, self.n_agents // 10):

            threshold_factor = getattr(self.args, 'key_node_threshold_factor', 1.2)
            max_key_ratio = 0.2
        else:

            threshold_factor = getattr(self.args, 'key_node_threshold_factor', 1.2)
            max_key_ratio = 0.25

        # 动态阈值
        min_threshold = mean_score + 0.5 * std_score
        max_threshold = mean_score + 2.0 * std_score
        threshold = mean_score + threshold_factor * std_score
        threshold = torch.clamp(threshold, min_threshold, max_threshold)

        # identify key_node
        key_indices = (scores > threshold).nonzero(as_tuple=True)[0].tolist()

        # based team_size
        max_key_nodes = max(1, int(n_nodes * max_key_ratio))
        if len(key_indices) > max_key_nodes:
            # 只保留分数最高的max_key_nodes个
            sorted_indices = torch.argsort(scores, descending=True)
            key_indices = sorted_indices[:max_key_nodes].tolist()

        if len(key_indices) == 0:
            key_indices = [torch.argmax(scores).item()]

        return key_indices

    def adjust_communication_safe(self, comm_output, key_node_info, sets, adjustment_type='intra'):

        if adjustment_type != 'intra' or not key_node_info['intra_key_nodes']:
            return comm_output.clone()

        adjusted_output = comm_output.detach().clone()
        n_agents = comm_output.shape[0]

        key_node_ratio = len(key_node_info['intra_key_nodes']) / n_agents

        enhancement_factor = 1.0 + self.comm_weight_enhance_factor * (1.0 - key_node_ratio)
        enhancement_factor = min(enhancement_factor, 1.5)

        key_indices = torch.tensor(key_node_info['intra_key_nodes'], device=comm_output.device)
        if len(key_indices) > 0:

            for idx in key_indices:
                if idx < n_agents:

                    original = comm_output[idx]
                    enhanced = original * enhancement_factor

                    adjusted_output[idx] = 0.7 * original + 0.3 * enhanced

        for team_set in sets:
            team_key_nodes = [idx for idx in key_node_info['intra_key_nodes'] if idx in team_set]
            team_non_key_nodes = [idx for idx in team_set if idx not in team_key_nodes]

            if not team_key_nodes or not team_non_key_nodes:
                continue

            key_weights = torch.ones(len(team_key_nodes), device=comm_output.device)
            key_weights = key_weights / key_weights.sum()
            key_repr = torch.zeros_like(comm_output[0])

            for w, idx in zip(key_weights, team_key_nodes):
                key_repr = key_repr + w * comm_output[idx]

            for node_idx in team_non_key_nodes:
                if node_idx < n_agents:
                    # The absorption rate decreases over time.
                    absorb_ratio = 0.2 / len(team_key_nodes)

                    adjusted_output[node_idx] = (1 - absorb_ratio) * comm_output[node_idx] + absorb_ratio * key_repr


        output_norm = torch.norm(adjusted_output, dim=1, keepdim=True) + 1e-8
        input_norm = torch.norm(comm_output, dim=1, keepdim=True) + 1e-8
        scale_factors = input_norm / output_norm
        scale_factors = torch.clamp(scale_factors, 0.8, 1.2)

        adjusted_output = adjusted_output * scale_factors

        return adjusted_output


class AgentAC(nn.Module):
    def __init__(self, args):
        super(AgentAC, self).__init__()
        # 新增：关键节点相关的超参数
        self.key_node_use_inter_key = getattr(args, 'use_inter_key_nodes', True)
        self.key_node_use_intra_key = getattr(args, 'use_intra_key_nodes', True)
        self.key_node_team_adjust_coeff = getattr(args, 'team_adjustment_coeff', 0.1)
        self.intra_struct_weight = getattr(args, 'intra_struct_weight', 0.5)
        self.key_node_identify_threshold = getattr(args, 'key_node_identify_threshold', 1.0)


        self.args = args
        self.n_agents = args.n_agents
        self.hid_size = args.hid_size
        self.n_actions = self.args.n_actions
        self.tanh = nn.Tanh()
        self.att_head = self.args.att_head

        self.message_dim = 64

        self.fc_1 = nn.Linear(self.hid_size +  self.message_dim * 2 , self.hid_size)
        self.fc_2 = nn.Linear(self.hid_size, self.hid_size)
        self.actor_head = nn.Linear(self.hid_size, self.n_actions)
        self.value_head = nn.Linear(self.hid_size, 1)


        self.local_fc_emb = nn.Linear(self.args.obs_shape, self.hid_size)
        self.inter_fc_emb = nn.Linear(self.args.obs_shape, self.message_dim)
        self.intra_fc_emb = nn.Linear(self.args.obs_shape, self.message_dim)



        self.intra_attn_mu = nn.MultiheadAttention(self.message_dim, num_heads=self.att_head, batch_first=True)
        self.intra_attn_std = nn.MultiheadAttention(self.message_dim, num_heads=self.att_head, batch_first=True)

        self.inter_attn_mu = nn.MultiheadAttention(self.message_dim, num_heads=self.att_head, batch_first=True)
        self.inter_attn_std = nn.MultiheadAttention(self.message_dim, num_heads=self.att_head, batch_first=True)

        self.attset_fc = nn.Linear(self.message_dim, 1)



    def forward(self, final_obs):
        h = self.tanh(self.fc_1(final_obs))
        h = self.tanh(self.fc_2(h))
        a = F.log_softmax(self.actor_head(h), dim=-1)
        v = self.value_head(h)
        return a, v


    def local_emb(self, x):
        return self.tanh(self.local_fc_emb(x))

    def inter_emb(self, x):
        return self.tanh(self.inter_fc_emb(x))

    def intra_emb(self, x):
        return self.tanh(self.intra_fc_emb(x))



    def _adjust_team_communication(self, team_obs, key_node_mask, key_node_adjustment):

        if not key_node_mask.any():
            return team_obs

        adjusted_obs = team_obs.clone()

        if key_node_adjustment > 0:
            adjusted_obs[key_node_mask] = adjusted_obs[key_node_mask] * (1.0 + key_node_adjustment)

        return adjusted_obs

    def intra_com(self, input):
        x = input.unsqueeze(0)
        mu, intra_attn_weights = self.intra_attn_mu(x,x,x)
        std, _ = self.intra_attn_std(x,x,x)
        std = F.softplus(std.squeeze(0)-5, beta = 1)
        intra_obs = self.reparameterise(mu.squeeze(0), std)

        if intra_attn_weights.dim() == 3:
            intra_attn_weights = intra_attn_weights.mean(dim=0)  # [n_members, n_members]
        else:
            intra_attn_weights = intra_attn_weights
        return intra_obs, mu.squeeze(0), std, intra_attn_weights



    def inter_com(self, input):
        """Modify inter_com to increase the returned attention weights"""
        x = input.unsqueeze(0)  # [1, n_teams, message_dim]
        # Modify to obtain mu and attention weights
        mu, inter_attn_weights = self.inter_attn_mu(x, x, x, average_attn_weights=False)
        std, _ = self.inter_attn_std(x,x,x)
        std = F.softplus(std.squeeze(0)-5, beta = 1)
        # Handling the shape of attention weights
        if inter_attn_weights.dim() == 3:
            # Average attention weight of all heads
            inter_attn_weights = inter_attn_weights.mean(dim=0)  # [n_teams, n_teams]
        else:
            inter_attn_weights = inter_attn_weights
        inter_obs = self.reparameterise(mu.squeeze(0), std)
        return inter_obs, mu.squeeze(0), std, inter_attn_weights

    def pooling(self, input, key_node_mask=None, key_weight_factor=1.0):
        """
        Pooling method, supporting adjustment of key node weights
        """
        score = self.attset_fc(input)  # [n_members, 1]

        # If the key node masks are provided and the weight factor is not 1, adjust the scores of the key nodes.
        if key_node_mask is not None and key_weight_factor != 1.0:
            # Ensure that the mask shape is correct
            if key_node_mask.dim() == 1:
                key_node_mask = key_node_mask.unsqueeze(-1)  # [n_members, 1]

            # Create a copy of the score to avoid operating on the original one.
            score = score.clone()

            # Adjust the scores of the key nodes
            mask_expanded = key_node_mask.bool()
            if mask_expanded.any():
                score[mask_expanded] *= key_weight_factor

        score = F.softmax(score, dim=0)
        output = torch.sum(score * input, dim=0, keepdim=True)
        return output

    def reparameterise(self, mu, std):
        eps = torch.randn_like(std)
        return mu + std * eps




class Teaming(nn.Module):

    def __init__(self, args):
        super(Teaming, self).__init__()

        self.args = args

        self.n_agents = self.args.n_agents
        # === Dynamic calculation of the maximum team size ===
        if hasattr(args, 'max_group') and args.max_group > 0:
            # If max_group is specified in the configuration, use the configuration value.
            self.max_group = args.max_group
        else:
            # dynamic comput
            if self.n_agents <= 5:
                self.max_group = min(4, self.n_agents)
            elif self.n_agents <= 10:
                self.max_group = min(6, self.n_agents // 2)
            elif self.n_agents <= 20:
                self.max_group = min(8, self.n_agents // 3)
            else:
                self.max_group = min(10, self.n_agents // 4)

        self.max_group = max(2, self.max_group)
        self.max_group = min(self.max_group, self.n_agents)

        print(f"Teaming模块初始化: n_agents={self.n_agents}, max_group={self.max_group}")

        self.hid_size = self.args.hid_size

        self.tanh = nn.Tanh()

        self.group_diversity_weight = getattr(args, 'group_diversity_weight', 0.1)

        self.fc1 = nn.Linear(self.args.obs_shape * self.n_agents, self.hid_size * 2)
        self.fc2 = nn.Linear(self.hid_size * 2, self.hid_size)
        self.fc3 = nn.Linear(self.args.obs_shape, self.hid_size)
        self.fc4 = nn.Linear(self.hid_size * 2, self.hid_size)
        self.action_head = nn.Linear(self.hid_size, self.max_group)

        self.critic_fc1 = nn.Linear(self.args.obs_shape * self.n_agents, self.hid_size * 2)
        self.critic_fc2 = nn.Linear(self.hid_size * 2, self.hid_size)
        self.critic_fc3 = nn.Linear(self.n_agents, self.hid_size)
        self.critic_fc4 = nn.Linear(self.hid_size * 2, self.hid_size)
        self.value_head = nn.Linear(self.hid_size, 1)



    def forward(self, x):

        h = x.view(1, -1)
        h = self.tanh(self.fc1(h))
        z = self.tanh(self.fc2(h))
        x = self.tanh(self.fc3(x))
        xh = torch.cat([x, z.repeat(self.n_agents, 1)], dim=-1)
        xh= self.tanh(self.fc4(xh))

        raw_logits = self.action_head(xh)

        if self.training and self.group_diversity_weight > 0:
            # Calculate the probability of each team being selected
            team_probs = F.softmax(raw_logits, dim=-1)  # [n_agents, max_group]

            # The entropy allocated by the computing team
            team_entropy = -torch.sum(team_probs * torch.log(team_probs + 1e-8), dim=0).mean()

            #  logits to encourage diversity
            diversity_bonus = torch.randn_like(raw_logits) * 0.1 * self.group_diversity_weight
            raw_logits = raw_logits + diversity_bonus

        a = F.log_softmax(raw_logits, dim=-1)
        return a


    def critic(self, o, a):


        h = o.view(1, -1)
        h = self.tanh(self.critic_fc1(h))
        z = self.tanh(self.critic_fc2(h))

        a = torch.Tensor(np.array(a)).view(1, -1)
        a = self.tanh(self.critic_fc3(a))

        ha = torch.cat([z, a], dim=-1)
        ha = self.tanh(self.critic_fc4(ha))
        v = self.value_head(ha)
        return v