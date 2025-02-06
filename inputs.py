import time
from pynput import keyboard
import numpy as np

# Monster hunter
# INPUTS = np.array([
#     # Move
#     keyboard.KeyCode.from_char('w'),
#     keyboard.KeyCode.from_char('a'),
#     keyboard.KeyCode.from_char('s'),
#     keyboard.KeyCode.from_char('d'),

#     # Attacks
#     keyboard.KeyCode.from_char('i'),
#     keyboard.KeyCode.from_char('p'),
#     keyboard.Key.shift_l,
#     keyboard.Key.space,

#     # Camera
#     keyboard.KeyCode.from_char('o'),
#     keyboard.KeyCode.from_char('k'),
#     keyboard.KeyCode.from_char('l'),
#     keyboard.KeyCode.from_char(';'),

#     # Items
#     keyboard.KeyCode.from_char('1'),
#     keyboard.KeyCode.from_char('2'),
#     keyboard.KeyCode.from_char('3'),
#     keyboard.KeyCode.from_char('4'),

#     # Others
#     keyboard.KeyCode.from_char('r'), #run
#     keyboard.KeyCode.from_char('c'), # Re target
#     keyboard.Key.alt_l, # Recenter camera
# ])

#Rabbit and steel
INPUTS = np.array([
    keyboard.Key.left,
    keyboard.Key.right,
    keyboard.Key.up,
    keyboard.Key.down,
    keyboard.KeyCode.from_char('z'),
    keyboard.KeyCode.from_char('x'),
    keyboard.KeyCode.from_char('c'),
    keyboard.KeyCode.from_char('v'),
])

actions = np.zeros(len(INPUTS))
stop_action = False
toggle = False
input_board = keyboard.Controller()

def reset():
    global actions
    actions = np.zeros(len(INPUTS))
reset()


def on_press(key):
    if key in INPUTS:
        actions[np.where(INPUTS==key)[0]] = 1.0

def on_press_extra(key):
    global stop_action, toggle, actions
    if key == keyboard.Key.esc:
        exit()
    if key == keyboard.Key.enter:
        stop_action = True
    if key == keyboard.KeyCode.from_char('`'):
        toggle = not toggle
        print("toggle " + str(toggle))
        if not toggle:
            actions = np.zeros(len(INPUTS))
            for i in range(len(INPUTS)):
                input_board.release(INPUTS[i])


def on_release(key):
    if key in INPUTS:
        actions[np.where(INPUTS==key)[0]] = 0.0

listener = keyboard.Listener(on_press=on_press_extra)
listener.start()

def main(surpress = False):
    listener = keyboard.Listener(on_press=on_press, on_release=on_release, surpress=surpress)
    listener.start()

if __name__ == "__main__":
    main()
    while True:
        time.sleep(1)
        print(actions)
        board = keyboard.Controller()