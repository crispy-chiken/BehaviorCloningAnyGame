from win32gui import GetWindowText, GetForegroundWindow
from win32gui import FindWindow, GetWindowRect
import numpy as np
from matplotlib import pyplot as plt
from PIL import ImageGrab

def grab_screen(display = False):
    window_handle = FindWindow(None, GetWindowText(GetForegroundWindow()))
    window_rect   = GetWindowRect(window_handle)

    raw = np.array(ImageGrab.grab(bbox=window_rect))[:,:,:3].astype(np.uint8)
    if display:
        print(window_rect)
        plt.imshow(raw)
        plt.show()
    return raw

if __name__ == '__main__':
    grab_screen(True)
