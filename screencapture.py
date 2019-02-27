import numpy as np
import cv2
from PIL import ImageGrab as ig
import time
import pyautogui
from keypress import ReleaseKey, PressKey, LEFT, RIGHT


def draw_lines(img,lines):
    try:
        for line in lines:
            coords = line[0]
            cv2.line(img, (coords[0], coords[1]), (coords[2], coords[3]), [255,255,255], 3)
    except:
        pass

def roi(img, vertices):
    #blank mask:
    mask = np.zeros_like(img)
    # fill the mask
    cv2.fillPoly(mask, vertices, 255)
    # now only show the area that is the mask
    masked = cv2.bitwise_and(img, mask)
    return masked

def left():
    PressKey(LEFT)
    ReleaseKey(LEFT)

def right():
    PressKey(RIGHT)
    ReleaseKey(RIGHT)

for i in list(range(4))[::-1]:
        print(i+1)
        time.sleep(1)

pyautogui.click(x=410, y=610)

last_time = time.time()
while(True):
    #print('Loop took {} seconds',format(time.time()-last_time))
    #left()
    #right()
    screen = np.array(ig.grab(bbox=(0, 37, 708, 434)))
    #screen = np.array(ig.grab(bbox=(26, 160, 680, 590)))
    new_screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
    processed_screen= cv2.Canny(new_screen, threshold1=200, threshold2=300)
    processed_img = cv2.GaussianBlur(processed_screen,(5, 5),0)
    vertices = np.array([[0, 37],[708,37],[708, 434],[0,434],], np.int32)
    processed_img = roi(processed_img, [vertices])
    lines= cv2.HoughLinesP(processed_img, 1, np.pi/180, 180, 150, 200)
    draw_lines(processed_img, lines)
    cv2.imshow("test", processed_img)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break
