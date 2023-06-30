# -*- coding: utf-8 -*-
"""
Created on Fri Jun 23 13:41:35 2023

@author: slbouknight
"""

import numpy as np
from collections import namedtuple, deque

# Defining step
step = namedtuple('Step', ['state', 'action', 'reward', 'done'])

# Iterate through n steps (eligibilit trace)

class NStepProgress:
    
    def __init__(self, env, ai, n_step):
        self.ai = ai
        self.rewards = []
        self.env = env
        self.n_step = n_step
        
    def __iter__(self):
        state = self.env.reset()
        history = deque()
        reward = 0.0
        
        while True:
            action = self.ai(np.array([state]))[0][0]
            next_state, r, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            reward += r
            history.append(step(state=state, action=action, reward=r, done=done))
            
            while len(history) > self.n_step + 1:
                history.popleft()
                
            if len(history) == self.n_step + 1:
                yield tuple(history)
                
            state = next_state
            
            if done:
                if len(history) > self.n_step + 1:
                    history.popleft()
                    
                while len(history) >= 1:
                    yield tuple(history)
                    history.popleft()
                    
                self.rewards.append(reward)
                reward = 0.0
                state = self.env.reset()
                history.clear()
                
    def rewards_steps(self):
        rewards_steps = self.rewards
        self.rewards = []
        return rewards_steps
    
# Experience Replay

class ReplayMemory:
    def __init__(self, n_steps, capacity=10000):
        self.capacity = capacity
        self.n_steps = n_steps
        self.n_steps_iter = iter(n_steps)
        self.buffer = deque()
        
    # Creates Iterator that returns random batches
    def sample_batch(self, batch_size):
        ofs = 0
        vals = list(self.buffer)
        np.random.shuffle(vals)
        
        while(ofs+1)*batch_size <= len(self.buffer):
            yield vals[ofs*batch_size:(ofs+1)*batch_size]
            ofs += 1
            
    def run_steps(self, samples):
        while samples > 0:
            # 10 consecutive steps
            entry = next(self.n_steps_iter)
            
            # 200 for current episode
            self.buffer.append(entry)
            samples -= 1
            
        while len(self.buffer) > self.capacity:
            # Pop anything exceeding capacity
            self.buffer.popleft()
        