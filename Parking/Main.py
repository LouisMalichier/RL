# Environment
import gymnasium as gym

# Models and computation
import torch
# Visualization
from tqdm.notebook import trange

# IO
from tqdm.notebook import trange
from utils import record_videos, show_videos
import Model
from config_parking import config_parking

env = gym.make("parking-v0", config=config_parking)
data = Model.collect_interaction_data(env)
print("Sample transition:", data[0])

dynamics = Model.DynamicsModel(state_size=env.observation_space.spaces["observation"].shape[0],
                         action_size=env.action_space.shape[0],
                         hidden_size=64,
                         dt=1/env.unwrapped.config["policy_frequency"])
print("Forward initial model on a sample transition:", dynamics(data[0].state.unsqueeze(0),
                                                                data[0].action.unsqueeze(0)).detach())

# Split dataset into training and validation
train_ratio = 0.7
train_data, validation_data = data[:int(train_ratio * len(data))], \
                              data[int(train_ratio * len(data)):]

optimizer = torch.optim.Adam(dynamics.parameters(), lr=0.01)

Model.train(dynamics, data, validation_data, optimizer=optimizer)

Model.visualize_trajectories(dynamics, state=torch.Tensor([0, 0, 0, 0, 1, 0]))


obs, info = env.reset()

print("Reward of a sample transition:", Model.reward_model(torch.Tensor(obs["observation"]).unsqueeze(0),
                                                     torch.Tensor(obs["desired_goal"])))

# Run the planner on a sample transition
action = Model.cem_planner(torch.Tensor(obs["observation"]),
                     torch.Tensor(obs["desired_goal"]),
                     env.action_space.shape[0], 
                     dynamics_model=dynamics)
print("Planned action:", action)

env = gym.make("parking-v0", render_mode='rgb_array')
env = record_videos(env)
obs, info = env.reset()



for step in trange(3 * env.config["duration"], desc="Testing 3 episodes..."):
    action = Model.cem_planner(torch.Tensor(obs["observation"]),
                         torch.Tensor(obs["desired_goal"]),
                         env.action_space.shape[0], 
                         dynamics_model=dynamics)
    obs, reward, done, truncated, info = env.step(action.numpy())
    if done or truncated:
        obs, info = env.reset()
env.close()
show_videos()