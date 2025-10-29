#Importing libraries
import gymnasium as gym
import numpy as np
import crafter
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from collections import defaultdict
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import cv2

#Downsampling the image to size 16x16
def rgb_downsample(obs, target_hw=(16, 16)):
    #Transposing it into an environemnt which is suitable to use openCV
    obs = np.array(obs, dtype=np.float32)
    if obs.shape[0] == 3 and len(obs.shape) == 3:
        obs_img = np.transpose(obs, (1, 2, 0))
    else:
        obs_img = obs
        #Resizing the image
    obs_down = cv2.resize(obs_img, target_hw, interpolation=cv2.INTER_AREA)
    #Normalising the values
    obs_down = obs_down.astype(np.float32) / 255.0
    obs_down = np.transpose(obs_down, (2, 0, 1))
    return obs_down

#Creating a wrapper function
class CrafterDictWrapper(gym.Env):
    def __init__(self, view=(9, 9), image_hw=(16, 16), max_steps_anneal=1_000_000, min_factor=0.1):
        self.env = crafter.env.Env(view=view)
        obs_sample = self.env.reset()
        sample_img = rgb_downsample(obs_sample, target_hw=image_hw)
        self.inventory_keys = ['plant', 'stone', 'iron', 'coal', 'diamond', 'wood', 'key', 'sword']
        self.achievement_names = [
            'plant', 'stone', 'iron', 'coal', 'diamond', 'crafting_table',
            'wood', 'workbench', 'furnace', 'door', 'chest', 'open_chest',
            'key', 'open_door', 'sword', 'kill_monster', 'eat', 'drink', 'explore'
        ]
        inventory_size = len(self.inventory_keys)
        achievement_size = len(self.achievement_names)

        self.observation_space = gym.spaces.Dict({
            "image": gym.spaces.Box(0, 1, shape=sample_img.shape, dtype=np.float32),
            "inventory": gym.spaces.Box(0, 1, shape=(inventory_size,), dtype=np.float32),
            "achievements": gym.spaces.Box(0, 1, shape=(achievement_size,), dtype=np.float32),
        })
        self.action_space = gym.spaces.Discrete(self.env.action_space.n)
        self.prev_achievements = set()
        self.visited_positions = set()
        self.prev_inventory = {}
        self.image_hw = image_hw
        self.total_steps = 0
        self.max_steps_anneal = max_steps_anneal
        self.min_factor = min_factor

#Reset function
    def reset(self, *, seed=None, options=None):
        obs = self.env.reset()
        self.prev_achievements = set()
        self.visited_positions = set()
        self.prev_inventory = {k: 0 for k in self.inventory_keys}
        inventory_vec = np.zeros(len(self.inventory_keys), dtype=np.float32)
        achievements_vec = np.zeros(len(self.achievement_names), dtype=np.float32)
        image = rgb_downsample(obs, target_hw=self.image_hw)
        obs_dict = {
            "image": image,
            "inventory": inventory_vec,
            "achievements": achievements_vec
        }
        return obs_dict, {}
#Step function
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.total_steps += 1
        anneal_factor = max(self.min_factor, 1 - self.total_steps / self.max_steps_anneal)
        shaped_reward = reward

 #Tracking achievements
        if "achievements" in info:
            for k, v in info["achievements"].items():
                if v and k not in self.prev_achievements:
                    shaped_reward += anneal_factor * 1.0
                    self.prev_achievements.add(k)

#Exploration reward for visiting new position
        pos = (info.get("x", None), info.get("y", None))
        if None not in pos:
            if pos not in self.visited_positions:
                shaped_reward += anneal_factor * 0.4
                self.visited_positions.add(pos)

#Inventory growth reward
        if "inventory" in info:
            for k in self.inventory_keys:
                new_count = info["inventory"].get(k, 0)
                prev_count = self.prev_inventory.get(k, 0)

                #Comparing the current count to the previous count
                if new_count > prev_count:
                    shaped_reward += anneal_factor * 0.8 * (new_count - prev_count)
                self.prev_inventory[k] = new_count

        inventory_vec = np.array([info.get("inventory", {}).get(k, 0) / 100.0 for k in self.inventory_keys], dtype=np.float32)
        achievements_vec = np.array([1.0 if info.get("achievements", {}).get(a, False) else 0.0 for a in self.achievement_names], dtype=np.float32)
        image = rgb_downsample(obs, target_hw=self.image_hw)
        obs_dict = {
            "image": image,
            "inventory": inventory_vec,
            "achievements": achievements_vec
        }
        terminated = done
        truncated = False

        info['raw_reward'] = reward

        return obs_dict, shaped_reward, terminated, truncated, info

    def render(self, mode="human"):
        return self.env.render(mode=mode)
    
#Defining a custom feature extractor which uses a deep CNN for the image and concatenates the resulting features with the inventory and achievements folloowing which it passes everything through a fully connected layer.
class DeeperCNNLSTMFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=256):
        super().__init__(observation_space, features_dim)
        #Functions to obstain the shape of the image, observation and inventory vecors
        img_shape = observation_space['image'].shape
        inv_shape = observation_space['inventory'].shape[0]
        ach_shape = observation_space['achievements'].shape[0]
#Building a deep CNN with four 2D layers with an increasing number of channels and a RELU activation function after each layer.
        self.cnn = nn.Sequential(
            nn.Conv2d(img_shape[0], 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            n_flatten = self.cnn(torch.zeros(1, *img_shape)).shape[1]
            #Final linear layer
        self.linear = nn.Sequential(
            nn.Linear(n_flatten + inv_shape + ach_shape, features_dim),
            nn.ReLU()
        )

    def forward(self, obs):
        # Converts each tensor to float
        img = torch.as_tensor(obs['image'], dtype=torch.float32)
        inv = torch.as_tensor(obs['inventory'], dtype=torch.float32)
        ach = torch.as_tensor(obs['achievements'], dtype=torch.float32)
        # If the image lacks a batch dimension, it adds one
        if img.ndim == 3:
            img = img.unsqueeze(0)
        cnn_out = self.cnn(img)
        #Ensures the inventory and achieve,ent vectors have a batch dimension of one
        if inv.ndim == 1:
            inv = inv.unsqueeze(0)
        if ach.ndim == 1:
            ach = ach.unsqueeze(0)
            #Concatenates the CNN output, inventory and achievements so there is a single feature vector
        concat = torch.cat([cnn_out, inv, ach], dim=-1)
        #The combined vector is passed through the fully linear laye
        features = self.linear(concat)
        return features

#Increasing the number of environments to 16 for faster computation
if __name__ == "__main__":
    n_envs = 16
    single_env = CrafterDictWrapper(view=(9, 9), image_hw=(16, 16), max_steps_anneal=1_000_000, min_factor=0.1)
    achievement_names = single_env.achievement_names

    env = SubprocVecEnv([
        lambda: CrafterDictWrapper(view=(9, 9), image_hw=(16, 16), max_steps_anneal=1_000_000, min_factor=0.1)
        for _ in range(n_envs)
    ])
    eval_env = SubprocVecEnv([
        lambda: CrafterDictWrapper(view=(9, 9), image_hw=(16, 16), max_steps_anneal=1_000_000, min_factor=0.1)
        for _ in range(1)
    ])
#Combining a CNN feature extratcor with a LSTM with a hidden state size of dimension 128. The LSTM is shared between the policy and value function
    policy_kwargs = dict(
        features_extractor_class=DeeperCNNLSTMFeaturesExtractor,
        features_extractor_kwargs=dict(features_dim=128),
        lstm_hidden_size=128,
        shared_lstm=True,  
        enable_critic_lstm=False,  
        
)
    #Adding recurent PPO model
    model = RecurrentPPO(
        "MultiInputLstmPolicy",
        env,
        verbose=1,
        tensorboard_log="./ppo_crafter_tensorboard/",
        n_steps=4096,
        policy_kwargs=policy_kwargs
    )
    model.learn(total_timesteps=400000)
    model.save("ppo_crafter_lstm_cnn")


    achievement_counts = defaultdict(int)
    episode_achievement_counts = []
    episode_raw_rewards = []
    episode_lengths = []
    total_episodes = 0
    episode_achievement_rates = []

    n_eval_episodes = 400

    for ep in range(n_eval_episodes):
        #Resetting the environment at the start of each episode
        obs = eval_env.reset()
        done = [False]
        ep_raw_reward = 0
        ep_len = 0
        ep_achievements = defaultdict(int)
        lstm_states = None
        episode_start = np.ones((1,), dtype=bool)

#Looping while stepping through the environment until the episode ends
        while not done[0]:
            action, lstm_states = model.predict(obs, state=lstm_states, episode_start=episode_start)
            episode_start = np.zeros((1,), dtype=bool)
            obs, reward, done, info = eval_env.step(action)
            ep_raw_reward += info[0].get('raw_reward', 0)
            ep_len += 1

              #Tracking achievements
            if "achievements" in info[0]:
                for k, v in info[0]["achievements"].items():
                    if v:
                        ep_achievements[k] = 1
                        achievement_counts[k] += 1

                #Recording achivement counts, rewards, lengths, achievement rates  for each evalutaton episode          
        episode_achievement_counts.append(ep_achievements)
        episode_raw_rewards.append(ep_raw_reward)
        episode_lengths.append(ep_len)
        total_episodes += 1
        episode_rate = sum(ep_achievements.values()) / len(achievement_names)
        episode_achievement_rates.append(episode_rate)
#Calculating achivement unlock rates
    unlock_rates = {}
    for a in achievement_names:
        unlock_rates[a] = achievement_counts[a] / total_episodes if total_episodes > 0 else 0.0

    rates = np.array([unlock_rates[a] for a in achievement_names if unlock_rates[a] > 0])

    #Calculating geometric mean
    geom_mean = np.exp(np.log(rates).mean()) if len(rates) > 0 else 0.0

#Calculating average survivial time and cumulative rewrd
    avg_survival_time = np.mean(episode_lengths) if episode_lengths else 0
    avg_cumulative_raw_reward = np.mean(episode_raw_rewards) if episode_raw_rewards else 0


#Printing metrics
    print("Crafter Metrics Over {} Episodes:".format(total_episodes))
    print("Achievement unlock rates:")
    for a in achievement_names:
        print(f"  {a}: {unlock_rates[a]:.3f}")
    print(f"Geometric mean of unlock rates: {geom_mean:.3f}")
    print(f"Average survival time: {avg_survival_time:.2f} steps")
    print(f"Average cumulative RAW reward: {avg_cumulative_raw_reward:.2f}")

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, n_eval_episodes+1), episode_achievement_rates, marker='o', label='Achievement Rate')
    plt.title('Achievement Rate per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Achievement Rate')
    plt.grid(True)
    plt.tight_layout()
    plt.legend()
    plt.show()

#Plot of survivial time per episode
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, n_eval_episodes+1), episode_lengths, marker='o', color='orange', label='Survival Time')
    plt.title('Survival Time per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Survival Time (steps)')
    plt.grid(True)
    plt.tight_layout()
    plt.legend()
    plt.show()

#Plot of cumulative reward per episode
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, n_eval_episodes+1), episode_raw_rewards, marker='o', color='green', label='Cumulative RAW Reward')
    plt.title('Cumulative RAW Reward per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Cumulative RAW Reward')
    plt.grid(True)
    plt.tight_layout()
    plt.legend()
    plt.show()