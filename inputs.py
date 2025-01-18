import time
from pynput import keyboard
import numpy as np

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

def reset():
    global actions
    actions = np.zeros(len(INPUTS))
reset()


def on_press(key):
    if key in INPUTS:
        actions[np.where(INPUTS==key)[0]] = 1.0

def on_press_extra(key):
    global stop_action, toggle
    if key == keyboard.Key.esc:
        stop_action = True
    if key == keyboard.Key.space:
        toggle = not toggle
        if not toggle:
            actions = np.zeros(len(INPUTS))

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