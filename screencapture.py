import numpy as np
import cv2
from PIL import ImageGrab as ig
import time
import pyautogui
from keypress import ReleaseKey, PressKey, W


for i in list(range(4))[::-1]:
        print(i+1)
        time.sleep(1)

last_time = time.time()
while(True):
    #print('Loop took {} seconds',format(time.time()-last_time))
    PressKey(W)
    screen = np.array(ig.grab(bbox=(0, 37, 708, 434)))
    new_screen= cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)
    processed_screen= cv2.Canny(new_screen, threshold1=80, threshold2=100)
    cv2.imshow("test", new_screen)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break