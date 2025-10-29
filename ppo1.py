
#Imporing libraries including SubprocVecEnv in order to ensure that multiple environemnts can be run in parallel
import gymnasium as gym
import numpy as np
import crafter
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from collections import defaultdict
import matplotlib.pyplot as plt
import cv2

#Downsampling function in order to resize the image to 16x16 pixels
def rgb_downsample(obs, target_hw=(16, 16)):

    #Coverting to a NumPy array of floats
    obs = np.array(obs, dtype=np.float32)

    #Making sure that the dimensions are in the correct order when the image is resized
    if obs.shape[0] == 3 and len(obs.shape) == 3:
        obs_img = np.transpose(obs, (1, 2, 0))
    else:
        obs_img = obs

        #Using cv2 to resize the image
    obs_down = cv2.resize(obs_img, target_hw, interpolation=cv2.INTER_AREA)
    obs_down = obs_down.astype(np.float32) / 255.0
    obs_down = np.transpose(obs_down, (2, 0, 1))
    return obs_down

#Creating a wrapper function for the environment (which is inherited from gym.Env) so that it can be compatible with libraries such as Stable Baselines3 and Gymnasium
class CrafterDictWrapper(gym.Env):

    #Definiing the environment is partially observable
    def __init__(self, view=(9, 9), image_hw=(16, 16), max_steps_anneal=1_000_000, min_factor=0.1):
        self.env = crafter.env.Env(view=view)
        obs_sample = self.env.reset()

        #Processing a sample observation to define the observation space
        sample_img = rgb_downsample(obs_sample, target_hw=image_hw)
        self.inventory_keys = ['plant', 'stone', 'iron', 'coal', 'diamond', 'wood', 'key', 'sword']
        self.achievement_names = [
            'plant', 'stone', 'iron', 'coal', 'diamond', 'crafting_table',
            'wood', 'workbench', 'furnace', 'door', 'chest', 'open_chest',
            'key', 'open_door', 'sword', 'kill_monster', 'eat', 'drink', 'explore'
        ]

        self.observation_space = gym.spaces.Dict({
            "image": gym.spaces.Box(0, 1, shape=sample_img.shape, dtype=np.float32)
        })
        self.action_space = gym.spaces.Discrete(self.env.action_space.n)
        self.image_hw = image_hw
        self.total_steps = 0

#Reset function when starting a new episode. Note the resizing o the image
    def reset(self, *, seed=None, options=None):
        obs = self.env.reset()
        image = rgb_downsample(obs, target_hw=self.image_hw)
        obs_dict = {
            "image": image
        }
        return obs_dict, {}

#Defining the step function
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.total_steps += 1

    
        image = rgb_downsample(obs, target_hw=self.image_hw)
        obs_dict = {
            "image": image
        }
        terminated = done
        truncated = False

        info['raw_reward'] = reward

        return obs_dict, reward, terminated, truncated, info

    def render(self, mode="human"):
        return self.env.render(mode=mode)

#Setting the number of environments to 8 which can be run in parallel
if __name__ == "__main__":
    n_envs = 8
    #Creating a single instance in the Crafter Environment with a 9x9 image view and 16 x 16 image size
    single_env = CrafterDictWrapper(view=(9, 9), image_hw=(16, 16), max_steps_anneal=1_000_000, min_factor=0.1)
    achievement_names = single_env.achievement_names
#Creating a vectorised environment for training. A lambda function is required as SubprocVecEnv expects a list of functions that create new environments when called
    env = SubprocVecEnv([
        lambda: CrafterDictWrapper(view=(9, 9), image_hw=(16, 16), max_steps_anneal=1_000_000, min_factor=0.1)
        for _ in range(n_envs)
    ])
    #Creating a seperate vectorised environment for evaluation
    eval_env = SubprocVecEnv([
        lambda: CrafterDictWrapper(view=(9, 9), image_hw=(16, 16), max_steps_anneal=1_000_000, min_factor=0.1)
        for _ in range(1)
    ])
#Using a Multi input policy as both images and vectors are used.
    model = PPO("MultiInputPolicy", env, verbose=1, tensorboard_log="./ppo_crafter_tensorboard/", n_steps=1024)
    model.learn(total_timesteps=400000)
    model.save("ppo_crafter_base")

    #Initialising the variables
    achievement_counts = defaultdict(int)
    episode_achievement_counts = []
    episode_raw_rewards = []
    episode_lengths = []
    total_episodes = 0
    episode_achievement_rates = []

#Evaluating over 400 episodes
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

#Calculating the average unlock rate
    unlock_rates = {}
    for a in achievement_names:
        unlock_rates[a] = achievement_counts[a] / total_episodes if total_episodes > 0 else 0.0
#Calculating the geometric mean of unlock rates
    rates = np.array([unlock_rates[a] for a in achievement_names if unlock_rates[a] > 0])
    geom_mean = np.exp(np.log(rates).mean()) if len(rates) > 0 else 0.0

#Calculating teh average survival time as well as cumulative reward
    avg_survival_time = np.mean(episode_lengths) if episode_lengths else 0
    avg_cumulative_raw_reward = np.mean(episode_raw_rewards) if episode_raw_rewards else 0

#Metrics used for evaluation
    print("Crafter Metrics Over {} Episodes:".format(total_episodes))
    print("Achievement unlock rates:")
    for a in achievement_names:
        print(f"  {a}: {unlock_rates[a]:.3f}")
    print(f"Geometric mean of unlock rates: {geom_mean:.3f}")
    print(f"Average survival time: {avg_survival_time:.2f} steps")
    print(f"Average cumulative RAW reward: {avg_cumulative_raw_reward:.2f}")

#Plotting acievemnt rate per episode
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, n_eval_episodes+1), episode_achievement_rates, marker='o', label='Achievement Rate')
    plt.title('Achievement Rate per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Achievement Rate')
    plt.grid(True)
    plt.tight_layout()
    plt.legend()
    plt.show()
#Plotting survival time per episode
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, n_eval_episodes+1), episode_lengths, marker='o', color='orange', label='Survival Time')
    plt.title('Survival Time per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Survival Time (steps)')
    plt.grid(True)
    plt.tight_layout()
    plt.legend()
    plt.show()

#Plotting cumulative reward per episode
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, n_eval_episodes+1), episode_raw_rewards, marker='o', color='green', label='Cumulative RAW Reward')
    plt.title('Cumulative Reward per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Cumulative Reward')
    plt.grid(True)
    plt.tight_layout()
    plt.legend()
    plt.show()