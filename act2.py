
import gymnasium as gym
import numpy as np
import crafter
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import SubprocVecEnv
from collections import defaultdict
import matplotlib.pyplot as plt
import cv2
#Downsampling the image to 16x16 pixels
def rgb_downsample(obs, target_hw=(16, 16)):
    #Transposing it into an environemnt which is suitable to use openCV
    obs = np.array(obs, dtype=np.float32)
    if obs.shape[0] == 3 and len(obs.shape) == 3:
        # (3, H, W) -> (H, W, 3)
        obs_img = np.transpose(obs, (1, 2, 0))
    else:
        obs_img = obs
           #Using OpenCV to resize the image
    obs_down = cv2.resize(obs_img, target_hw, interpolation=cv2.INTER_AREA)
    obs_down = obs_down.astype(np.float32) / 255.0  # Normalize
    obs_down = np.transpose(obs_down, (2, 0, 1))    # (H, W, 3) -> (3, H, W)
    return obs_down

#Creating a wrapper for the environment (which is inherited from gym.Env) so that it can be compatible with libraries such as Stable Baselines3 and Gymnasium
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
#Adding inventory and achievements to the observation space
        self.observation_space = gym.spaces.Dict({
            "image": gym.spaces.Box(0, 1, shape=sample_img.shape, dtype=np.float32),
            "inventory": gym.spaces.Box(0, 1, shape=(inventory_size,), dtype=np.float32),
            "achievements": gym.spaces.Box(0, 1, shape=(achievement_size,), dtype=np.float32),
        })
        #Defining the actions that an agent can take 
        self.action_space = gym.spaces.Discrete(self.env.action_space.n)
         #Tracking state variables
        self.prev_achievements = set()
        self.visited_positions = set()
        self.prev_inventory = {}
        #Storing the height and width for the image
        self.image_hw = image_hw
        self.total_steps = 0
        self.max_steps_anneal = max_steps_anneal
        self.min_factor = min_factor

#Reset function when starting a new episodes.
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

#Updating the step function to include reward shaping
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.total_steps += 1
        anneal_factor = max(self.min_factor, 1 - self.total_steps / self.max_steps_anneal)
        shaped_reward = reward

#If a new achievement is unlocked, add a shaped reward
        if "achievements" in info:
            for k, v in info["achievements"].items():
                if v and k not in self.prev_achievements:
                    shaped_reward += anneal_factor * 1.0
                    self.prev_achievements.add(k)

#Shaped reward for exploration (0.3)
        pos = (info.get("x", None), info.get("y", None))
        if None not in pos:
            if pos not in self.visited_positions:
                shaped_reward += anneal_factor * 0.3
                self.visited_positions.add(pos)

#Shaped reward (0.6) for collecting new items
        if "inventory" in info:
            for k in self.inventory_keys:
                new_count = info["inventory"].get(k, 0)
                prev_count = self.prev_inventory.get(k, 0)
                if new_count > prev_count:
                    shaped_reward += anneal_factor * 0.6 * (new_count - prev_count)
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

if __name__ == "__main__":

    n_envs = 8  # Use more parallel environments
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

    model = A2C("MultiInputPolicy", env, verbose=1, tensorboard_log="./a2c_crafter_tensorboard/", n_steps=1024)
    model.learn(total_timesteps=400000)
    model.save("a2c_crafter_base")

#Metrics tracking
    achievement_counts = defaultdict(int)
    episode_achievement_counts = []
    episode_raw_rewards = []
    episode_lengths = []
    total_episodes = 0
    episode_achievement_rates = []

    n_eval_episodes = 400
    for ep in range(n_eval_episodes):
        obs = eval_env.reset()
        done = [False]
        ep_raw_reward = 0
        ep_len = 0
        ep_achievements = defaultdict(int)
        while not done[0]:
            action, _states = model.predict(obs)
            obs, reward, done, info = eval_env.step(action)
            ep_raw_reward += info[0].get('raw_reward', 0)
            ep_len += 1
            if "achievements" in info[0]:
                for k, v in info[0]["achievements"].items():
                    if v:
                        ep_achievements[k] = 1
                        achievement_counts[k] += 1
        episode_achievement_counts.append(ep_achievements)
        episode_raw_rewards.append(ep_raw_reward)
        episode_lengths.append(ep_len)
        total_episodes += 1
        episode_rate = sum(ep_achievements.values()) / len(achievement_names)
        episode_achievement_rates.append(episode_rate)

#Calculating average achievemnt unlock rates, geometric mean, avergae survival time and eaverage cumulative reward
    unlock_rates = {}
    for a in achievement_names:
        unlock_rates[a] = achievement_counts[a] / total_episodes if total_episodes > 0 else 0.0

    rates = np.array([unlock_rates[a] for a in achievement_names if unlock_rates[a] > 0])
    geom_mean = np.exp(np.log(rates).mean()) if len(rates) > 0 else 0.0

    avg_survival_time = np.mean(episode_lengths) if episode_lengths else 0
    avg_cumulative_raw_reward = np.mean(episode_raw_rewards) if episode_raw_rewards else 0

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

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, n_eval_episodes+1), episode_lengths, marker='o', color='orange', label='Survival Time')
    plt.title('Survival Time per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Survival Time (steps)')
    plt.grid(True)
    plt.tight_layout()
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, n_eval_episodes+1), episode_raw_rewards, marker='o', color='green', label='Cumulative RAW Reward')
    plt.title('Cumulative RAW Reward per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Cumulative RAW Reward')
    plt.grid(True)
    plt.tight_layout()
    plt.legend()
    plt.show()