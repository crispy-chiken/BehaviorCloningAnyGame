from __future__ import print_function

import gzip
import os
import pickle
import time
import inputs
from inputs import INPUTS
from train import data_transform, Net, DATA_DIR, MODEL_FILE
from pyglet.window import key
from environment import Game
import os
import torch

import gymnasium as gym
import numpy as np

DATA_DIR = 'data'
DATA_FILE = 'data.gzip'

def rollout(env:gym.Env, model:Net):
    global restart_train, agent_action, exit_train, pause_train
    agent_action = np.zeros(len(INPUTS))
    exit_train = False
    pause_train = False
    restart_train = False

    # if the file exists, append
    if os.path.exists(os.path.join(DATA_DIR, DATA_FILE)):
        with gzip.open(os.path.join(DATA_DIR, DATA_FILE), 'rb') as f:
            observations = pickle.load(f)
    else:
        observations = list()

    # Run key reading tool
    inputs.main()
    state = env.reset()
    total_reward = 0
    episode = 1
    inputs.toggle = True
    while 1:
        done = False
        env.render()

        if inputs.toggle:
            # Bot
           # _state = np.moveaxis(state, 2, 0)  # channel first image
            _state = state
            # numpy to tensor
            _state = torch.from_numpy(np.flip(_state, axis=0).copy())
            _state = data_transform(state)  # apply transformations
            _state = _state.unsqueeze(0)  # add additional dimension
            # forward
            with torch.set_grad_enabled(False):
                outputs = model(_state)[0]

            state, _, done, _ = env.step(outputs)  # one step
            # Clear actions 
            inputs.reset()
        else:
            agent_action = inputs.actions        
            a = np.copy(agent_action)
            old_state = state

            state, reward, done, info = env.step(agent_action)

            observations.append((old_state, a, state, reward, done))

            total_reward += reward

        if done:
                
            if episode == 1:
                # store generated data
                data_file_path = os.path.join(DATA_DIR, DATA_FILE)
                print("Saving observations to " + data_file_path)

                if not os.path.exists(DATA_DIR):
                    os.mkdir(DATA_DIR)

                with gzip.open(data_file_path, 'wb') as f:
                    pickle.dump(observations, f)
                
                env.close()
                return

            print("Episodes %i reward %0.2f" % (episode, total_reward))

            episode += 1

            state = env.reset()

        while pause_train:
            env.render()
            time.sleep(0.1)


if __name__ == '__main__':
    m = Net()
    m.load_state_dict(torch.load(os.path.join(DATA_DIR, MODEL_FILE)))
    print("loaded")
    m.eval()
    env = Game()
    env.reset()
    #env.render()
    rollout(env, m)
