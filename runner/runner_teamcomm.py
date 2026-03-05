import torch
import torch.nn as nn
from collections import namedtuple
import numpy as np
from torch.optim import Adam,RMSprop
from modules.utils import merge_dict, multinomials_log_densities,multinomials_log_density
import time
import random
from runner import Runner
import torch.nn.functional as F
from scipy.sparse import coo_matrix

import argparse

Transition      = namedtuple('Transition',      ('action_outs', 'actions', 'rewards', 'values',
                                                 'episode_masks', 'episode_agent_masks',
                                                 'mu', 'std', 'key_node_info'
                                                 ))
Team_Transition = namedtuple('Team_Transition', ('team_action_outs', 'team_actions', 'global_reward',
                                                 'global_value', 'episode_masks','score', 'key_node_info'))

class RunnerTeamComm(Runner):
    def __init__(self, config, env, agent):
        super().__init__(config, env, agent)

        self.args = argparse.Namespace(**config)

        self.n_teams = self.agent.teaming.max_group  # 动态获取

        self.optimizer_agent_ac = RMSprop(self.agent.agent.parameters(), lr=self.args.lr, alpha=0.97, eps=1e-6)
        self.optimizer_team_ac = RMSprop(self.agent.teaming.parameters(), lr=self.args.lr, alpha=0.97, eps=1e-6)

        self.n_nodes = int(self.n_agents * (self.n_agents - 1) / 2)
        self.interval = self.args.interval

        # 关键节点相关参数
        self.intra_key_coeff = getattr(self.args, 'intra_key_coeff', 0.1)
        self.inter_key_coeff = getattr(self.args, 'inter_key_coeff', 0.1)
        self.team_adjust_coeff = getattr(self.args, 'team_adjust_coeff', 0.05)

        self.use_key_node_for_vib = getattr(self.args, 'use_key_node_for_vib', True)
        self.key_node_vib_weight = getattr(self.args, 'key_node_vib_weight', 0.6)
        # 记录池化模式
        self.pooling_mode = getattr(self.args, 'pooling_mode', 'original')
        print(f"Runner使用池化模式: {self.pooling_mode}")

    def optimizer_zero_grad(self):
        self.optimizer_agent_ac.zero_grad()
        self.optimizer_team_ac.zero_grad()

    def optimizer_step(self):
        self.optimizer_agent_ac.step()
        self.optimizer_team_ac.step()

    def compute_grad(self, batch):
        log = dict()
        agent_log = self.compute_agent_grad(batch[0])
        team_log = self.compute_team_grad(batch[1])

        merge_dict(agent_log, log)
        merge_dict(team_log, log)
        return log

    def train_batch(self, batch_size):
        batch_data, batch_log = self.collect_batch_data(batch_size)
        self.optimizer_zero_grad()
        train_log = self.compute_grad(batch_data)
        merge_dict(batch_log, train_log)
        for p in self.params:
            if p._grad is not None:
                p._grad.data /= batch_log['num_steps']
        self.optimizer_step()
        return train_log

    def collect_batch_data(self, batch_size):
        agent_batch_data = []
        team_batch_data = []
        batch_log = dict()
        num_episodes = 0

        while len(agent_batch_data) < batch_size:
            episode_data, episode_log = self.run_an_episode()
            agent_batch_data += episode_data[0]
            team_batch_data += episode_data[1]
            merge_dict(episode_log, batch_log)
            num_episodes += 1

        batch_data = Transition(*zip(*agent_batch_data))
        team_batch_data = Team_Transition(*zip(*team_batch_data))
        batch_data = [batch_data, team_batch_data]
        batch_log['num_episodes'] = num_episodes
        batch_log['num_steps'] = len(batch_data[0].actions)

        return batch_data, batch_log

    def run_an_episode(self):

        log = dict()

        memory = []
        team_memory = []

        self.reset()
        obs = self.env.get_obs()

        obs_tensor = torch.tensor(np.array(obs), dtype=torch.float)
        team_action_out = self.agent.teaming(obs_tensor)
        team_action = self.choose_action(team_action_out)

        rewards_list = []
        global_reward = np.zeros(1)

        step = 1
        num_group = 0
        episode_return = 0
        done = False


        while not done and step <= self.args.episode_length:

            obs_tensor = torch.tensor(np.array(obs), dtype=torch.float)
            if step % self.interval == 0:
                # Regularly recalculate the team grouping
                team_action_out = self.agent.teaming(obs_tensor)
                team_action = self.choose_action(team_action_out)

            sets = self.matrix_to_set(team_action)
            score = self.compute_modularity_score(sets, obs_tensor)
            after_comm, mu, std, key_node_info = self.agent.communicate(
                obs_tensor,  #
                sets,
                team_action_out,
                step
            )
            action_outs, values = self.agent.agent(after_comm)
            actions = self.choose_action(action_outs)
            team_value = self.agent.teaming.critic(obs_tensor, actions)
            rewards, done, env_info = self.env.step(actions)

            rewards_list.append(np.mean(rewards).reshape(1))

            if step % self.interval == 0:
                global_reward = np.mean(rewards_list).reshape(1)

            next_obs = self.env.get_obs()
            done = done or step == self.args.episode_length

            episode_mask = np.ones(rewards.shape)
            episode_agent_mask = np.ones(rewards.shape)
            global_episode_mask = np.ones(1)
            if done:
                episode_mask = np.zeros(rewards.shape)
                global_episode_mask = np.zeros(1)
            elif 'completed_agent' in env_info:
                episode_agent_mask = 1 - np.array(env_info['completed_agent']).reshape(-1)

            trans = Transition(action_outs, actions, rewards, values, episode_mask, episode_agent_mask, mu, std, key_node_info)
            memory.append(trans)

            if step % self.interval == 0:
                team_trans = Team_Transition(team_action_out, team_action, global_reward, team_value,
                                             global_episode_mask,score, key_node_info)
                team_memory.append(team_trans)

            obs = next_obs
            episode_return += int(np.sum(rewards))
            step += 1
            num_group += len(sets)

        log['episode_return'] = episode_return
        log['episode_steps'] = [step - 1]
        log['num_groups'] = num_group / (step - 1)

        if 'num_collisions' in env_info:
            log['num_collisions'] = int(env_info['num_collisions'])

        if self.args.env == 'tj':
            merge_dict(self.env.get_stat(),log)

        return (memory, team_memory), log




    def compute_team_grad(self, batch):
        log = {}
        batch_size = len(batch.global_value)
        # self.n_teams = 4 #self.n_agents
        self.n_teams = self.agent.teaming.max_group

        rewards = torch.tensor(np.array(batch.global_reward))
        actions = torch.tensor(np.array(batch.team_actions)).transpose(1, 2).view(-1, 1, self.n_agents)
        episode_masks = torch.tensor(np.array(batch.episode_masks))
        values = torch.cat(batch.global_value, dim=0).view(batch_size, 1)
        score = torch.Tensor(np.array(batch.score)).view(batch_size, 1)

        action_outs = list(zip(*batch.team_action_outs))
        action_outs = [torch.cat(a, dim=0) for a in action_outs]

        # Obtain key node information
        key_node_info_list = batch.key_node_info
        if self.args.moduarity:
            modularity_loss = self._compute_team_modularity_loss(score, actions,action_outs)
            total_loss = modularity_loss
            log['moduarity_loss'] = modularity_loss.item()
            log['team_action_loss'] = modularity_loss.item()
            log['team_value_loss'] = modularity_loss.item()
            log['team_total_loss'] = modularity_loss.item()


        else:
            returns, advantages = self._compute_team_returns_advantages(rewards, values, episode_masks)
            actor_loss = self._compute_team_actor_loss(actions, advantages, action_outs)
            critic_loss = self._compute_team_critic_loss(values, returns)
            total_loss = actor_loss + self.args.value_coeff * critic_loss
            log['team_action_loss'] = actor_loss.item()
            log['team_value_loss'] = critic_loss.item()
            log['team_total_loss'] = total_loss.item()

        total_loss.backward()
        return log


    def _compute_team_returns_advantages(self, rewards, values, episode_masks):
        batch_size = rewards.size(0)
        returns = torch.empty(batch_size, 1)
        advantages = torch.empty(batch_size, 1)
        prev_returns = torch.zeros(1)

        # Reward normalization
        if self.args.normalize_rewards:
            mean_reward = rewards.mean()
            std_reward = rewards.std() + 1e-8  # Add a small value to prevent division by zero
            rewards = (rewards - mean_reward) / std_reward

        for i in reversed(range(batch_size)):
            returns[i] = rewards[i] + self.args.gamma * prev_returns * episode_masks[i]
            prev_returns = returns[i].clone()
            advantages[i] = returns[i] - values.data[i]

        # Normalizing advantages as well
        if self.args.normalize_advantages:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return returns, advantages


    def _compute_team_modularity_loss(self, score, actions, action_outs):
        log_p_a = [action_outs[i].view(-1, self.n_teams) for i in range(self.n_agents)]
        actions = actions.contiguous().view(-1, self.n_agents)
        log_prob = multinomials_log_densities(actions, log_p_a)

        mean_score = score.mean()
        std_score = score.std() + 1e-8  # Add a small value to prevent division by zero
        score = (score - mean_score) / std_score

        q_loss = - score.view(-1).unsqueeze(-1) * log_prob

        # Add team allocation diversity and encouragement
        if hasattr(self.args, 'group_diversity_weight') and self.args.group_diversity_weight > 0:
            # The entropy allocated by the computing team
            team_probs = [torch.exp(log_p_a[i]) for i in range(self.n_agents)]
            team_entropy = sum([-torch.sum(p * torch.log(p + 1e-8)) for p in team_probs]) / self.n_agents
            # Encourage high entropy (diversity)
            diversity_bonus = team_entropy * self.args.group_diversity_weight
            q_loss = q_loss - diversity_bonus.view(-1, 1)

        return q_loss.sum()




    def _compute_team_actor_loss(self, actions, advantages, action_outs):
        log_p_a = [action_outs[i].view(-1, self.n_teams) for i in range(self.n_agents)]
        actions = actions.contiguous().view(-1, self.n_agents)
        log_prob = multinomials_log_densities(actions, log_p_a)
        action_loss = -advantages.view(-1).unsqueeze(-1) * log_prob
        return action_loss.sum()

    def _compute_team_critic_loss(self, values, returns):
        return nn.functional.mse_loss(values.view(-1), returns.view(-1), reduction='sum')

    def compute_agent_grad(self, batch):
        log = dict()
        n = self.n_agents
        batch_size = len(batch.actions)

        rewards = torch.Tensor(np.array(batch.rewards))
        actions = torch.Tensor(np.array(batch.actions)).transpose(1, 2).view(-1, n, 1)
        episode_masks = torch.Tensor(np.array(batch.episode_masks))
        episode_agent_masks = torch.Tensor(np.array(batch.episode_agent_masks))
        values = torch.cat(batch.values, dim=0).view(batch_size, n)
        action_outs = torch.stack(batch.action_outs, dim=0)

        mus = torch.cat(batch.mu, dim=0)#.view(batch_size, n, -1) shape: [batch_size * n_agents, feature_dim]
        stds = torch.cat(batch.std, dim=0)#.view(batch_size, n, -1)
        key_node_info_list = batch.key_node_info
        # print(f"DEBUG: batch_size: {batch_size}")
        # print(f"DEBUG: mus shape: {mus.shape}")
        # print(f"DEBUG: stds shape: {stds.shape}")
        # print(f"DEBUG: key_node_info_list length: {len(key_node_info_list)}")

        returns, advantages = self._compute_returns_advantages(rewards, values, episode_masks, episode_agent_masks)

        actor_loss = self._compute_actor_loss(actions, advantages, action_outs)
        critic_loss = self._compute_critic_loss(values, returns)


        if self.args.vib:
            # Pass the key_node_info_list to the VIB loss calculation
            # Pass the correct batch_size and key_node_info_list to the VIB loss calculation
            vib_loss = self._compute_vib_loss(action_outs, mus, stds, key_node_info_list, batch_size)
            total_loss = actor_loss + self.args.value_coeff * critic_loss + self.args.vib_coeff * vib_loss

            log['action_loss'] = actor_loss.item()
            log['value_loss'] = critic_loss.item()
            log['vib_loss'] = vib_loss.item()
            log['total_loss'] = total_loss.item()

        else:
            total_loss = actor_loss + self.args.value_coeff * critic_loss
            log['action_loss'] = actor_loss.item()
            log['value_loss'] = critic_loss.item()
            log['total_loss'] = total_loss.item()

        total_loss.backward()
        return log



    def _compute_vib_loss(self, action_outs, mu, std, key_node_info_list=None, batch_size=None):
        # Calculate the cross-entropy loss
        log_p_a = action_outs.view(-1, self.n_actions)
        ce_loss = - (log_p_a * log_p_a.exp()).mean()

        if not self.use_key_node_for_vib or key_node_info_list is None or len(key_node_info_list) == 0:

            KL_loss = mu.pow(2) + std.pow(2) - 2 * torch.log(std + 1e-8) - 1
            weighted_KL = KL_loss.mean() * 0.05
            vib_loss = ce_loss + weighted_KL
            return vib_loss

        # mu shape: [batch_size * n_agents, feature_dim]
        #batch_size = mu.shape[0] / n_agents
        total_samples = mu.shape[0]
        calculated_batch_size = total_samples // self.n_agents

        # Verify
        if calculated_batch_size * self.n_agents != total_samples:
            calculated_batch_size = max(1, total_samples // self.n_agents)

        batch_size = calculated_batch_size
        feature_dim = mu.shape[-1] if mu.dim() > 1 else 1

        mu_reshaped = mu.view(batch_size, self.n_agents, feature_dim)
        std_reshaped = std.view(batch_size, self.n_agents, feature_dim)

        KL_per_agent = mu_reshaped.pow(2) + std_reshaped.pow(2) - 2 * torch.log(std_reshaped + 1e-8) - 1
        KL_per_agent = KL_per_agent.mean(dim=2)  # [batch_size, n_agents]
        # ini KL
        total_KL = 0.0
        total_weight = 0.0

        for i in range(min(calculated_batch_size, len(key_node_info_list))):
            key_node_info = key_node_info_list[i]

            # 4.1 Obtain the key node masks
            if key_node_info is None:

                if self.use_key_node_for_vib:

                    print(f"Warning: The time step {i} lacks key node information. Use standard weights.")
                    weight_matrix = torch.ones(self.n_agents, device=mu.device) * 0.05
                else:

                    weight_matrix = torch.ones(self.n_agents, device=mu.device) * 0.05
            elif 'key_node_mask' not in key_node_info:
                if 'intra_key_nodes' in key_node_info and len(key_node_info['intra_key_nodes']) > 0:
                    key_mask = torch.zeros(self.n_agents, device=mu.device)
                    for idx in key_node_info['intra_key_nodes']:
                        if idx < self.n_agents:
                            key_mask[idx] = 1.0

                    weight_matrix = torch.where(
                        key_mask > 0.5,
                        torch.tensor(self.key_node_vib_weight, device=mu.device),
                        torch.tensor(0.05, device=mu.device)
                    )
                else:

                    weight_matrix = torch.ones(self.n_agents, device=mu.device) * 0.05
            else:
                key_mask = key_node_info['key_node_mask']

                # makesure shape
                if len(key_mask) != self.n_agents:
                    if len(key_mask) < self.n_agents:
                        #
                        padded_mask = torch.zeros(self.n_agents, device=mu.device)
                        padded_mask[:len(key_mask)] = key_mask
                        key_mask = padded_mask
                    else:
                        key_mask = key_mask[:self.n_agents]


                weight_matrix = torch.where(
                    key_mask > 0.5,
                    torch.tensor(self.key_node_vib_weight, device=mu.device),
                    torch.tensor(0.05, device=mu.device)
                )

            for j in range(self.n_agents):
                weight = weight_matrix[j]
                total_KL += KL_per_agent[i, j] * weight
                total_weight += weight

        if total_weight > 0:
            weighted_KL = total_KL / total_weight
        else:

            KL_loss = mu.pow(2) + std.pow(2) - 2 * torch.log(std + 1e-8) - 1
            weighted_KL = KL_loss.mean() * 0.05

        return ce_loss + weighted_KL





    def compute_modularity_score(self, sets, obs_tensor, gamma = 1.0):
        # Compute similarity matrix
        similarity_matrix = self.cosine_similarity_matrix(obs_tensor).cpu().numpy()

        # Compute the modularity score
        m = np.sum(similarity_matrix) / 2.0

        score = 0
        for agent_set in sets:
            k_i = np.sum(similarity_matrix[agent_set, :])
            sum_of_edges = np.sum(similarity_matrix[np.ix_(agent_set, agent_set)])
            delta = np.zeros_like(similarity_matrix)
            delta[np.ix_(agent_set, agent_set)] = 1.0
            score += sum_of_edges - gamma * (k_i ** 2) / (2 * m) * np.sum(delta)

        score = score / (2 * m)

        n_groups = len(sets)
        n_agents = self.n_agents

        if n_agents <= 5:
            ideal_groups = max(2, n_agents // 2)
        elif n_agents <= 10:
            ideal_groups = max(3, n_agents // 3)
        else:
            ideal_groups = max(4, n_agents // 4)

        group_penalty = abs(n_groups - ideal_groups) / max(ideal_groups, 1)
        group_constraint = 1.0 - min(group_penalty, 0.7)

        constrained_score = score * group_constraint

        return constrained_score

    def cosine_similarity_matrix(self, obs):
        """
        obs: [n_agents, obs_dim] as a PyTorch tensor
        Returns a matrix of size [n_agents, n_agents] with the cosine similarity between rows.
        """
        norm = obs.norm(p=2, dim=1, keepdim=True)
        obs_normalized = obs.div(norm)

        similarity_matrix = torch.mm(obs_normalized, obs_normalized.t())

        # Set diagonal to zero
        similarity_matrix.fill_diagonal_(0)

        return similarity_matrix









