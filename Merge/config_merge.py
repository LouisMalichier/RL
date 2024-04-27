import gymnasium as gym

env_merge = gym.make("merge-v0", render_mode="rgb_array")

config_merge = {
    
        "collision_reward": -1,
        "right_lane_reward": 0.1,
        "high_speed_reward": 0.2,
        "reward_speed_range": [20, 30],
        "merging_speed_reward": -0.5,
        "lane_change_reward": -0.05,
}


env_merge.unwrapped.configure(config_merge)
print(env_merge.reset())