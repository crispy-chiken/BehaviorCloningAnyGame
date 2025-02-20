from __future__ import print_function

import os
import gymnasium as gym
import numpy as np
import torch

from matplotlib import pyplot as plt
import cv2
from train import data_transform, Net, DATA_DIR, MODEL_FILE
from pyglet.window import key
from gymnasium.spaces.box import Box
from environment import Game
from inputs import INPUTS
import inputs

def play(model):
    """
    Let the agent play
    :param model: the network
    """
    env:gym.Env = Game()

    # initialize environment
    state = env.reset()
    
    #env.unwrapped.viewer.window.on_key_press = key_press
    while True:
        #env.render()
        # plt.imshow(state)
        # plt.show(block=False)
        # plt.draw()
        state = np.moveaxis(state, 2, 0)  # channel first image

        # numpy to tensor
        state = torch.from_numpy(np.flip(state, axis=0).copy())
        state = data_transform(state)  # apply transformations
        state = state.unsqueeze(0)  # add additional dimension

        # forward
        with torch.set_grad_enabled(False):
            outputs = model(state)[0]

        # normalized = torch.nn.functional.softmax(outputs, dim=1)

        # # translate from net output to env action
        # max_action = np.argmax(normalized.cpu().numpy()[0])
        # action = INPUTS[max_action]
        
        state, _, terminal, _ = env.step(outputs)  # one step

        if terminal:
            return

if __name__ == '__main__':
    m = Net()
    m.load_state_dict(torch.load(os.path.join(DATA_DIR, MODEL_FILE)))
    print("loaded")
    m.eval()
    # # Run key reading tool
    # inputs.main()
    play(m)
