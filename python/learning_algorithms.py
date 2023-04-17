import pickle
import gymnasium as gym
import torch
from gymnasium.utils.save_video import save_video
from torch import nn
from torch.optim import Adam
from torch.distributions import Categorical
from utils import *
import tensorflow as tf


# Class for training an RL agent within an environment
class PGTrainer:
    def __init__(self, params):
        self.params = params
        self.env = gym.make(self.params['env_name'])
        self.agent = Agent(env=self.env, params=self.params)
        self.actor_policy = PGPolicy(input_size=self.env.observation_space.shape[0], output_size=self.env.action_space.n, hidden_dim=self.params['hidden_dim']).to(get_device())
        self.optimizer = Adam(params=self.actor_policy.parameters(), lr=self.params['lr'])

    def run_training_loop(self):
        list_ro_reward = list()

        for ro_idx in range(self.params['n_rollout']):
            trajectory = self.agent.collect_trajectory(policy=self.actor_policy)
            loss = self.estimate_loss_function(trajectory)
            self.update_policy(loss)
            # TODO: Calculate avg reward for this rollout
            # HINT: Add all the rewards from each trajectory. There should be "ntr" trajectories within a single rollout.
            avg_ro_reward = 0
            for t in range(len(trajectory['reward'])):
                avg_ro_reward = avg_ro_reward + np.sum(r for r in trajectory['reward'][t])
            avg_ro_reward = avg_ro_reward / self.params['n_trajectory_per_rollout']
            print(f'End of rollout {ro_idx}: Average trajectory reward is {avg_ro_reward: 0.2f}')
            # Append average rollout reward into a list
            list_ro_reward.append(avg_ro_reward)
        # Save avg-rewards as pickle files
        pkl_file_name = self.params['exp_name'] + '.pkl'
        with open(pkl_file_name, 'wb') as f:
            pickle.dump(list_ro_reward, f)
        # Save a video of the trained agent playing
        self.generate_video()
        # Close environment
        self.env.close()

    def estimate_loss_function(self, trajectory):
        loss = list()
        for t_idx in range(self.params['n_trajectory_per_rollout']):
            # TODO: Compute loss function
            # HINT 1: You should implement eq 6, 7 and 8 here. Which will be used based on the flags set from the main function
            # HINT 2: Get trajectory action log-prob
            # HINT 3: Calculate Loss function and append to the list

            log_prob = trajectory['log_prob'][t_idx]
            returns_t = apply_return(trajectory['reward'][t_idx])
            reward_to_go = apply_reward_to_go(trajectory['reward'][t_idx])
            discount_reward = apply_discount(trajectory['reward'][t_idx])
            
            reward_for_trajectory = 0

            for t in range(len(log_prob) - 1):
                t_value = log_prob[t]

                if self.params['reward_to_go']:
                    t_value = t_value * reward_to_go[t]
                elif self.params['reward_discount']:
                    t_value = t_value * discount_reward[t]

                reward_for_trajectory = reward_for_trajectory + t_value
                
            if not(self.params['reward_to_go']) and not(self.params['reward_discount']):
                reward_for_trajectory = reward_for_trajectory * returns_t[0]

            loss.append( -1 * reward_for_trajectory )
        loss = torch.stack(loss).mean()
        return loss

    def update_policy(self, loss):
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        
    def generate_video(self, max_frame=1000):
        self.env = gym.make(self.params['env_name'], render_mode='rgb_array_list')
        obs, _ = self.env.reset()
        for _ in range(max_frame):
            action_idx, log_prob = self.actor_policy(torch.tensor(obs, dtype=torch.float32, device=get_device()))
            obs, reward, terminated, truncated, info = self.env.step(self.agent.action_space[action_idx.item()])
            if terminated or truncated:
                break
        save_video(frames=self.env.render(), video_folder=self.params['env_name'][:-3], fps=self.env.metadata['render_fps'], step_starting_index=0, episode_index=0)

    

# CLass for policy-net
class PGPolicy(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim):
        super(PGPolicy, self).__init__()

        # TODO: Define the policy net
        # HINT: You can use nn.Sequential to set up a 2 layer feedforward neural network.
        self.policy_net = nn.Sequential(nn.Linear(input_size, hidden_dim),nn.Linear(hidden_dim, output_size))
        

    def forward(self, obs):
        # TODO: Forward pass of policy net
        # HINT: (use Categorical from torch.distributions to draw samples and log-prob from model output)
        action_probs = self.policy_net(obs)
        probability_distribution = Categorical(action_probs)
        action_index = probability_distribution.sample()
        log_prob = probability_distribution.log_prob(action_index)
        return action_index, log_prob

    def forwardpolicy(self, obs):
        # TODO: Forward pass of policy net
        # HINT: (use Categorical from torch.distributions to draw samples and log-prob from model output)
        obs_tensor = torch.from_numpy(obs)
        action_logits = self.policy_net(obs_tensor)
        action_probs = torch.nn.functional.softmax(action_logits, dim=-1)
        probability_distribution = Categorical(action_probs)
        action_index = probability_distribution.sample()
        log_prob = probability_distribution.log_prob(action_index)
        return action_index, log_prob


# Class for agent
class Agent:
    def __init__(self, env, params=None):
        self.env = env
        self.params = params
        self.action_space = [action for action in range(self.env.action_space.n)]

    def collect_trajectory(self, policy):
        obs, _ = self.env.reset(seed=self.params['rng_seed'])
        rollout_buffer = list()
        for _ in range(self.params['n_trajectory_per_rollout']):
            trajectory_buffer = {'log_prob': list(), 'reward': list()}
            while True:
                # TODO: Get action from the policy (forward pass of policy net)
                action_idx, log_prob =policy.forwardpolicy(obs)

                # TODO: Step environment (use self.env.step() function)
                obs, reward, terminated, truncated, info = self.env.step(self.action_space[action_idx.item()])
                
                # Save log-prob and reward into the buffer
                trajectory_buffer['log_prob'].append(log_prob)
                trajectory_buffer['reward'].append(reward)
                
                # Check for termination criteria
                if terminated or truncated:
                    obs, _ = self.env.reset()
                    rollout_buffer.append(trajectory_buffer)
                    break
        rollout_buffer = self.serialize_trajectory(rollout_buffer)
        return rollout_buffer

    # Convreward_for_trajectorys a list-of-dictionary into dictionary-of-list
    @staticmethod
    def serialize_trajectory(rollout_buffer):
        serialized_buffer = {'log_prob': list(), 'reward': list()}
        for trajectory_buffer in rollout_buffer:
            serialized_buffer['log_prob'].append(torch.stack(trajectory_buffer['log_prob']))
            serialized_buffer['reward'].append(trajectory_buffer['reward'])
        return serialized_buffer
