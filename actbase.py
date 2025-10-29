
#Importing libraries including Crafter and Gymnasium
import crafter
import gymnasium as gym
from stable_baselines3 import A2C
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt

#Defining a new class which inherits from gym.Env to create a wrapper for the Crafter environment
class CrafterGymWrapper(gym.Env):
    def __init__(self):
        self.env = crafter.env.Env()
        #Creates an instance of the underlying environment
        space = self.env.observation_space
        #Defining the observation space for the wrapper class
        self.observation_space = gym.spaces.Box(
            low=space.low,
            high=space.high,
            shape=tuple(int(v) for v in space.shape),
            dtype=space.dtype
        )
        #Defining the action space for the wrapper
        self.action_space = gym.spaces.Discrete(self.env.action_space.n)
#Resets the environment to start a new episode
    def reset(self, *, seed=None, options=None):
        obs = self.env.reset()
        return obs, {}
#Advances the environment by one step based on the action taken
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return obs, reward, done, False, info
#Renders the environment for visualisation
    def render(self, mode="human"):
        return self.env.render(mode=mode)


#Initialising the environment so that it is partially observable
env = CrafterGymWrapper()
model = A2C("CnnPolicy", env, verbose=1, tensorboard_log="./a2c_crafter_tensorboard/")
model.learn(total_timesteps=400000)
model.save("a2c_crafter_base")

# Initialising the variables to track metrics while the agent is interacting with the environment
achievement_names = [
    'plant', 'stone', 'iron', 'coal', 'diamond', 'crafting_table',
    'wood', 'workbench', 'furnace', 'door', 'chest', 'open_chest',
    'key', 'open_door', 'sword', 'kill_monster', 'eat', 'drink', 'explore'
]
achievement_counts = defaultdict(int)
episode_achievement_counts = []
episode_rewards = []
episode_lengths = []
episode_achievement_rates = []  # <-- Per-episode achievement rate
total_episodes = 0

#Evaluating over 400 episodes
n_eval_episodes = 400
for ep in range(n_eval_episodes):
    #Reset the environment at the start of each episode
    obs, _ = env.reset()
    done = False
    truncated = False
    ep_reward = 0
    ep_len = 0
    ep_achievements = defaultdict(int)
    while not done:
         #Using the trained model to pick an action based on the current observation
        action, _states = model.predict(obs)
        obs, reward, done, truncated, info = env.step(action)
        ep_reward += reward
        ep_len += 1
        if "achievements" in info:
            for k, v in info["achievements"].items():
                if v:
                    ep_achievements[k] = 1
                    achievement_counts[k] += 1
        if done:
            break
              #Storing the results
    episode_achievement_counts.append(ep_achievements)
    episode_rewards.append(ep_reward)
    episode_lengths.append(ep_len)
    total_episodes += 1

   # Calculate achievement rate for this episode
    episode_rate = sum(ep_achievements.values()) / len(achievement_names)
    episode_achievement_rates.append(episode_rate)

# Achievement unlock rate
unlock_rates = {}
for a in achievement_names:
    unlock_rates[a] = achievement_counts[a] / total_episodes if total_episodes > 0 else 0.0

# Geometric mean of achievement unlock rates
rates = np.array([unlock_rates[a] for a in achievement_names if a in unlock_rates and unlock_rates[a] > 0])
geom_mean = np.exp(np.log(rates).mean()) if len(rates) > 0 else 0.0

# Survival time
avg_survival_time = np.mean(episode_lengths) if episode_lengths else 0

# Cumulative reward
avg_cumulative_reward = np.mean(episode_rewards) if episode_rewards else 0

#Printing the metrics
print("Crafter Metrics Over {} Episodes:".format(total_episodes))
print("Achievement unlock rates:")
for a in achievement_names:
    print(f"  {a}: {unlock_rates[a]:.3f}")
print(f"Geometric mean of unlock rates: {geom_mean:.3f}")
print(f"Average survival time: {avg_survival_time:.2f} steps")
print(f"Average cumulative reward: {avg_cumulative_reward:.2f}")

#Visualisation

#  Line graph for achievement rate per episode ---
plt.figure(figsize=(10, 5))
plt.plot(range(1, n_eval_episodes+1), episode_achievement_rates, marker='o', label='Achievement Rate')
plt.title('Achievement Rate per Episode')
plt.xlabel('Episode')
plt.ylabel('Achievement Rate')
plt.grid(True)
plt.tight_layout()
plt.legend()
plt.show()

#  Line graph for survival time per episode ---
plt.figure(figsize=(10, 5))
plt.plot(range(1, n_eval_episodes+1), episode_lengths, marker='o', color='orange', label='Survival Time')
plt.title('Survival Time per Episode')
plt.xlabel('Episode')
plt.ylabel('Survival Time (steps)')
plt.grid(True)
plt.tight_layout()
plt.legend()
plt.show()

#  Line graph for return over episodes ---
plt.figure(figsize=(10, 5))
plt.plot(range(1, n_eval_episodes+1), episode_rewards, marker='o', color='green', label='Cumulative Reward')
plt.title('Cumulative Reward per Episode')
plt.xlabel('Episode')
plt.ylabel('Cumulative Reward')
plt.grid(True)
plt.tight_layout()
plt.legend()
plt.show()