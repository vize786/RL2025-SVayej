
#Installing libraries
import gymnasium as gym
import numpy as np
import crafter
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import SubprocVecEnv
from collections import defaultdict
import matplotlib.pyplot as plt
import cv2


#Function to take an image, downsample it to a reduced size (16x16 pixels), normalise and reshape it 
def rgb_downsample(obs, target_hw=(16, 16)):
    obs = np.array(obs, dtype=np.float32)

    
    #Checking if the observations are channel first. If they are channel first, they need to be transposed to rearrange the axis
    if obs.shape[0] == 3 and len(obs.shape) == 3:
        obs_img = np.transpose(obs, (1, 2, 0))
    else:
        obs_img = obs
    obs_down = cv2.resize(obs_img, target_hw, interpolation=cv2.INTER_AREA)
    obs_down = obs_down.astype(np.float32) / 255.0
    obs_down = np.transpose(obs_down, (2, 0, 1))
    return obs_down


#Creating a wrapper for the environment
class CrafterDictWrapper(gym.Env):
    def __init__(self, view=(9, 9), image_hw=(16, 16), max_steps_anneal=1_000_000, min_factor=0.1):
        self.env = crafter.env.Env(view=view)
        obs_sample = self.env.reset()

        #Processes the sample observation image
        sample_img = rgb_downsample(obs_sample, target_hw=image_hw)
        self.inventory_keys = ['plant', 'stone', 'iron', 'coal', 'diamond', 'wood', 'key', 'sword']
        self.achievement_names = [
            'plant', 'stone', 'iron', 'coal', 'diamond', 'crafting_table',
            'wood', 'workbench', 'furnace', 'door', 'chest', 'open_chest',
            'key', 'open_door', 'sword', 'kill_monster', 'eat', 'drink', 'explore'
        ]
        inventory_size = len(self.inventory_keys)
        achievement_size = len(self.achievement_names)


        #Defining the shape and type of observations that the environment will return. Normalising the image, inventory and achievements
        self.observation_space = gym.spaces.Dict({
            "image": gym.spaces.Box(0, 1, shape=sample_img.shape, dtype=np.float32),
            "inventory": gym.spaces.Box(0, 1, shape=(inventory_size,), dtype=np.float32),
            "achievements": gym.spaces.Box(0, 1, shape=(achievement_size,), dtype=np.float32),
        })

               #Setting the action space, tarckers, image processing parameters, and step counters
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

        #Creating a numpy array with float length equal to the number of inventory keys
        inventory_vec = np.zeros(len(self.inventory_keys), dtype=np.float32)

        
          #Creating a numpy array with float length equal to the number of achievements
        achievements_vec = np.zeros(len(self.achievement_names), dtype=np.float32)
        image = rgb_downsample(obs, target_hw=self.image_hw)

        #Packaging the processed image, observation and achievemnts into a single dictionary. The dictionary forms the observation that will be returned to the environment
        obs_dict = {
            "image": image,
            "inventory": inventory_vec,
            "achievements": achievements_vec
        }
        return obs_dict, {}


#Step function with annealed rewards
    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        #Running count of steps taken 
        self.total_steps += 1

           #Calculating an anneling factor which reduces shaped rewards as the training increases 
        anneal_factor = max(self.min_factor, 1 - self.total_steps / self.max_steps_anneal)
        shaped_reward = reward

         #Reward of +1 for new achievements
        if "achievements" in info:

            #Checking to see if the achievement is currently unlocked or have we recordeed ir previously
            for k, v in info["achievements"].items():
                if v and k not in self.prev_achievements:
                    shaped_reward += anneal_factor * 1.0
                    self.prev_achievements.add(k)


             #Positions which have been visited: Reward of +0.4        
        pos = (info.get("x", None), info.get("y", None))

         #Checking to see if the position is valid and if it is new
        if None not in pos:
            if pos not in self.visited_positions:
                shaped_reward += anneal_factor * 0.3
                self.visited_positions.add(pos)

                 #Inventory increases- reward of +0.8  
        if "inventory" in info:
            for k in self.inventory_keys:
                new_count = info["inventory"].get(k, 0)
                prev_count = self.prev_inventory.get(k, 0)

                 #If the agent collected more items in this check than the previous, it is givem a shaped reward
                if new_count > prev_count:
                    shaped_reward += anneal_factor * 0.6 * (new_count - prev_count)
                self.prev_inventory[k] = new_count

        inventory_vec = np.array([info.get("inventory", {}).get(k, 0) / 100.0 for k in self.inventory_keys], dtype=np.float32)
        achievements_vec = np.array([1.0 if info.get("achievements", {}).get(a, False) else 0.0 for a in self.achievement_names], dtype=np.float32)

            #Downsampling and normalising the image
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

#Setting the number of environments to 16

if __name__ == "__main__":
    n_envs = 16

        #Creating a single instance in the Crafter Environment with a 9x9 image view and 16 x 16 image size
    single_env = CrafterDictWrapper(view=(9, 9), image_hw=(16, 16), max_steps_anneal=1_000_000, min_factor=0.1)
    achievement_names = single_env.achievement_names
    inventory_keys = single_env.inventory_keys


    env = SubprocVecEnv([
        lambda: CrafterDictWrapper(view=(9, 9), image_hw=(16, 16), max_steps_anneal=1_000_000, min_factor=0.1)
        for _ in range(n_envs)
    ])

    eval_env = SubprocVecEnv([
        lambda: CrafterDictWrapper(view=(9, 9), image_hw=(16, 16), max_steps_anneal=1_000_000, min_factor=0.1)
        for _ in range(1)
    ])


#Building the model with 1 500 000 timesteps
    model = A2C("MultiInputPolicy", env, verbose=1, tensorboard_log="./a2c_crafter_tensorboard/", n_steps=2048)
    model.learn(total_timesteps=1500000)
    model.save("a2c_crafter_base")



    achievement_counts = defaultdict(int)
    episode_achievement_counts = []
    episode_raw_rewards = []
    episode_lengths = []
    total_episodes = 0
    episode_achievement_rates = []

    # Tracking the unique places which the agent visits dring a single episode
    episode_unique_positions = []
    WORLD_SIZE = 64
   


    # Recording the overall inventory map during an episode
    episode_inventory_growth = []
    episode_inventory_growth_per_item = {k: [] for k in inventory_keys}

    # Listing the action entropy and stagnation metrics
    episode_action_entropy = []
    episode_stagnation_counts = []



#Number of Evalutaion episodes
    n_eval_episodes = 1500
    for ep in range(n_eval_episodes):

        obs = eval_env.res  #Resetting the environment at the start of each episdoeet()
        done = [False]
        ep_raw_reward = 0
        ep_len = 0
        ep_achievements = defaultdict(int)
        visited_positions = set()
        inventory_growth = 0
        inventory_growth_item = {k: 0 for k in inventory_keys}
        last_inventory = {k: 0 for k in inventory_keys}

     # Lists to store tracking values for action, entropies and stagnation count
        actions = []
        entropies = []
        stagnation_count = 0
        prev_action = None
        consecutive_repeat = 0

#Looping while stepping through the environment until the episdoe ends
        while not done[0]:
            # Get action and action probabilities from model
            action, _states = model.predict(obs, deterministic=False)
            actions.append(action[0] if isinstance(action, np.ndarray) else action)

             #Obtaining the action distribution, probabilities and entropy. This is done by converting the observation into a PyTorch tensor suitable for the policy
            import torch

                   #Obtaining the action distribution as well as the probabilities
            with torch.no_grad():
                obs_tensor = {k: torch.tensor(v, dtype=torch.float32).unsqueeze(0) for k, v in obs.items()} if isinstance(obs, dict) else obs
                dist = model.policy.get_distribution(obs_tensor)
                probs = dist.distribution.probs.cpu().numpy()[0]
                entropy = -np.sum(probs * np.log(probs + 1e-8))
                entropies.append(entropy)
            obs, reward, done, info = eval_env.step(action)

                  #Accumulate rewards
            ep_raw_reward += info[0].get('raw_reward', 0)
            ep_len += 1

                #Tracking the visited positions 
            pos = (info[0].get("x", None), info[0].get("y", None))
            if None not in pos:
                visited_positions.add(pos)

                       #Tracking achievements
            if "achievements" in info[0]:
                for k, v in info[0]["achievements"].items():
                    if v:
                        ep_achievements[k] = 1
                        achievement_counts[k] += 1

                             #Tracking inventory growth
            if "inventory" in info[0]:
                for k in inventory_keys:
                    new_count = info[0]["inventory"].get(k, 0)
                    growth = new_count - last_inventory[k]
                    if growth > 0:
                        inventory_growth += growth
                        inventory_growth_item[k] += growth
                    last_inventory[k] = new_count
            # Stagnation detections
            current_action = action[0] if isinstance(action, np.ndarray) else action

   #Checking if the current action is the same as the previous action
            if prev_action is not None and current_action == prev_action:
                consecutive_repeat += 1
            else:
                consecutive_repeat = 0

                #Update the previous action to the current action
            prev_action = current_action

            #If the agent has repeated the same action for 10 or more consecutive steps, there is stagnation
            if consecutive_repeat >= 10:  
                stagnation_count += 1


#Recording achivement counts, rewards, lengths, achievement rates and other metrics for each evalutaton episode
        episode_achievement_counts.append(ep_achievements)
        episode_raw_rewards.append(ep_raw_reward)
        episode_lengths.append(ep_len)
        total_episodes += 1
        episode_rate = sum(ep_achievements.values()) / len(achievement_names)
        episode_achievement_rates.append(episode_rate)
        episode_unique_positions.append(len(visited_positions))
        episode_inventory_growth.append(inventory_growth)


         #Looping over every inventory key in order to record growth per item
        for k in inventory_keys:
            episode_inventory_growth_per_item[k].append(inventory_growth_item[k])
        # Entropy and stagnation metrics
        episode_action_entropy.append(np.mean(entropies) if entropies else 0)
        episode_stagnation_counts.append(stagnation_count)


#Computing achievement unlock rates
    unlock_rates = {}
    for a in achievement_names:
        unlock_rates[a] = achievement_counts[a] / total_episodes if total_episodes > 0 else 0.0

        #Calculating metrics
    rates = np.array([unlock_rates[a] for a in achievement_names if unlock_rates[a] > 0])
    geom_mean = np.exp(np.log(rates).mean()) if len(rates) > 0 else 0.0
    avg_survival_time = np.mean(episode_lengths) if episode_lengths else 0
    avg_cumulative_raw_reward = np.mean(episode_raw_rewards) if episode_raw_rewards else 0
    avg_unique_positions = np.mean(episode_unique_positions) if episode_unique_positions else 0
    avg_inventory_growth = np.mean(episode_inventory_growth) if episode_inventory_growth else 0
    avg_entropy = np.mean(episode_action_entropy) if episode_action_entropy else 0
    avg_stagnation = np.mean(episode_stagnation_counts) if episode_stagnation_counts else 0


#Printing metrics
    print("Crafter Metrics Over {} Episodes:".format(total_episodes))
    print("Achievement unlock rates:")
    for a in achievement_names:
        print(f"  {a}: {unlock_rates[a]:.3f}")
    print(f"Geometric mean of unlock rates: {geom_mean:.3f}")
    print(f"Average survival time: {avg_survival_time:.2f} steps")
    print(f"Average cumulative RAW reward: {avg_cumulative_raw_reward:.2f}")
    print(f"Average unique positions visited (exploration): {avg_unique_positions:.2f}")
    print(f"Average inventory growth (items collected): {avg_inventory_growth:.2f}")
    print("Average inventory growth per item:")


    for k in inventory_keys:
        avg_item_growth = np.mean(episode_inventory_growth_per_item[k]) if episode_inventory_growth_per_item[k] else 0
        print(f"  {k}: {avg_item_growth:.2f}")
    print(f"Average action entropy: {avg_entropy:.4f}")
    print(f"Average stagnation events per episode: {avg_stagnation:.2f}")

    #Plot of policy improvement
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, n_eval_episodes+1), episode_raw_rewards, marker='o', color='green', label='Cumulative RAW Reward')
    plt.plot(range(1, n_eval_episodes+1), episode_unique_positions, marker='o', color='purple', label='Unique Positions Visited')
    plt.title('Policy Improvement Over Time')
    plt.xlabel('Episode')
    plt.ylabel('Metric Value')
    plt.grid(True)
    plt.tight_layout()
    plt.legend()
    plt.show()

    # Plot of inventory growth
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, n_eval_episodes+1), episode_inventory_growth, marker='o', color='brown', label='Total Inventory Growth')
    for k in inventory_keys:
        plt.plot(range(1, n_eval_episodes+1), episode_inventory_growth_per_item[k], marker='.', linestyle='--', label=f'{k} Growth')
    plt.title('Inventory Growth per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Items Collected')
    plt.grid(True)
    plt.tight_layout()
    plt.legend()
    plt.show()

    # Plot of action entropy
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, n_eval_episodes+1), episode_action_entropy, marker='o', color='magenta', label='Action Entropy')
    plt.title('Action Entropy per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Mean Action Entropy')
    plt.grid(True)
    plt.tight_layout()
    plt.legend()
    plt.show()

    # Plot of stagnation
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, n_eval_episodes+1), episode_stagnation_counts, marker='o', color='red', label='Stagnation Events')
    plt.title('Stagnation Events per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Stagnation Events (>=10 repeats)')
    plt.grid(True)
    plt.tight_layout()
    plt.legend()
    plt.show()

