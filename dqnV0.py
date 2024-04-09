import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

#ALPHA is learning rate
class DeepQNetwork(nn.Module):
    def __init__(self, ALPHA, input_dims, fc1_dims, fc2_dims,
                 n_actions):
        super(DeepQNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, self.n_actions)
        self.optimizer = optim.Adam(self.parameters(), lr=ALPHA)
        self.loss = nn.MSELoss()
        #self.device = T.device('cuda:0' if T.cuda.is_available() else 'cuda:1')
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, observation):
        #print("type(obs)", observation.dtype)
        #print("obs", observation)
        #if observation.dtype == object : print(observation)
        state = T.Tensor(observation).to(self.device)
        #observation = observation.view(-1)
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        actions = self.fc3(x)
        print("actions", actions)
        return actions

class Agent(object):
    def __init__(self, gamma, epsilon, alpha, input_dims, batch_size, n_actions,
                 max_mem_size=4, eps_end=0.01, eps_dec=0.996):
        self.GAMMA = gamma
        self.EPSILON = epsilon
        self.EPS_MIN = eps_end
        self.EPS_DEC = eps_dec
        self.ALPHA = alpha
        self.action_space = [i for i in range(n_actions)]
        self.n_actions = n_actions
        self.mem_size = max_mem_size
        self.batch_size = batch_size
        self.mem_cntr = 0
        self.Q_eval = DeepQNetwork(alpha, n_actions=self.n_actions,
                              input_dims=input_dims, fc1_dims=256, fc2_dims=256)
        #self.state_memory = np.zeros((self.mem_size, *input_dims))
        self.state_memory = np.zeros((self.mem_size, 2), dtype=object)
        #self.new_state_memory = np.zeros((self.mem_size, *input_dims))

        self.new_state_memory = np.zeros((self.mem_size, *input_dims), dtype=object)
        self.action_memory = np.zeros((self.mem_size, self.n_actions),
                                      dtype=np.uint8)
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.uint8)

    def storeTransition(self, state, action, reward, state_, terminal):
        index = self.mem_cntr % self.mem_size
        #print("mem_size", self.mem_size)
        #print("mem_count", self.mem_cntr)
        #print("index", index)

        # Assign the NumPy array to one part of self.state_memory
        self.state_memory[index, 0] = state[0]
        print("State 0 : Obs", state[0])

        # Assign the dictionary to another part of self.state_memory (e.g., as metadata)
        self.state_memory[index, 1] = state[1]
        #print("state_memory ", self.state_memory)

        #self.state_memory[index] = state
        actions = np.zeros(self.n_actions)
        actions[action] = 1.0
        self.action_memory[index] = actions
        self.reward_memory[index] = reward

        self.new_state_memory[index] = tuple(state_)
        self.terminal_memory[index] = 1 - terminal
        self.mem_cntr += 1

    def chooseAction(self, observation):
        rand = np.random.random()
        #("rand", rand)
        actions = self.Q_eval.forward(observation)
        #print("actions", actions)
        #print("argmax", T.argmax(actions))
        if rand > self.EPSILON:
            action = T.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)
        return action

    def learn(self):
        if self.mem_cntr > self.batch_size:
            self.Q_eval.optimizer.zero_grad()

            max_mem = self.mem_cntr if self.mem_cntr < self.mem_size \
                                    else self.mem_size

            batch = np.random.choice(max_mem, self.batch_size)
            #print("batch nb", batch)
            state_batch = self.state_memory[batch[0], 0]
            print("state_batch", state_batch)
            action_batch = self.action_memory[batch]
            action_values = np.array(self.action_space, dtype=np.int32)
            action_indices = np.dot(action_batch, action_values)
            reward_batch = self.reward_memory[batch]
            new_state_batch = self.new_state_memory[batch[0], 0]
            #print("new_state_batch", new_state_batch)
            terminal_batch = self.terminal_memory[batch]

            reward_batch = T.Tensor(reward_batch).to(self.Q_eval.device)
            terminal_batch = T.Tensor(terminal_batch).to(self.Q_eval.device)
            print("terminal_batch", terminal_batch)

            q_eval = self.Q_eval.forward(state_batch).to(self.Q_eval.device)
            #q_target = self.Q_eval.forward(state_batch).to(self.Q_eval.device)
            q_target = q_eval.clone()
            print("q_target", q_target)
            q_next = self.Q_eval.forward(new_state_batch).to(self.Q_eval.device)
            print("q_next", q_next)

            batch_index = np.arange(self.batch_size, dtype=np.int32)
            print("type batch_index", type(batch_index), batch_index)
            print("type action_indices", type(action_indices), action_indices)
            print("batch_index[0], action_indices[0]", batch_index[0], action_indices[0])
            #print("q_target[0,0] ", q_target[0])
            #rint("T.max(q_next, dim=0)[0]", T.max(q_next, dim=0)[0])
            print("reward_batch + self.GAMMA*T.max(q_next, dim=0)[0]*terminal_batch", reward_batch + self.GAMMA*T.max(q_next, dim=0)[0]*terminal_batch)
            print("q_target[batch_index]", q_target[batch_index])
            q_target[batch_index] = reward_batch + self.GAMMA*T.max(q_next, dim=0)[0]*terminal_batch

            self.EPSILON = self.EPSILON*self.EPS_DEC if self.EPSILON > \
                           self.EPS_MIN else self.EPS_MIN

            loss = self.Q_eval.loss(q_target, q_eval).to(self.Q_eval.device)
            loss.backward()
            self.Q_eval.optimizer.step()