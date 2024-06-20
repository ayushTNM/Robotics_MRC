#!/usr/bin/env python3


import sys
import termios
import time
import tty
from time import sleep

import cv2
import numpy as np
import picar_4wd as fc
from picamera2 import Picamera2

print('Please run under desktop environment (eg: vnc) to display the image window')

kernel_5 = np.ones((5,5),np.uint8)
#global object_detection_bool
#object_detection_bool = True
#Define a 5×5 convolution kernel with element values of all 1.
power_val = 10 #default reg - 0-4
color_dict = {'red':[0,4],'orange':[5,18],'yellow':[22,37],'green':[42,85],'blue':[92,110],'purple':[115,165],'red_2':[165,180]}  #Here is the range of H in the HSV color space represented by the color
key = 'status'
print("If you want to quit.Please press q")

def readchar():
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch

def readkey(getchar_fn=None):
    getchar = getchar_fn or readchar
    c1 = getchar()
    if ord(c1) != 0x1b:
        return c1
    c2 = getchar()
    if ord(c2) != 0x5b:
        return c1
    c3 = getchar()
    return chr(0x10 + ord(c3) - 65)

def performInput(key, camera = None):
    print(power_val)
    fc.stop()
    img = None
    if not camera is None:
        time.sleep(0.1)
        img = camera.capture_array()
    if key == 119: # w
        fc.forward(power_val)
    elif key == 97: # a
        fc.turn_left(power_val)
    elif key == 115: # s
        fc.backward(power_val)
    elif key == 100: # d
        fc.turn_right(power_val)
    elif key == 32: #space
        fc.stop()
    return img

def getOpposite(input):
    if input == 119: # w
        return 115 # s
    elif input == 97: # a
        return 100 # d
    elif input == 115: # s
        return 119 # w
    elif input == 100: # d
        return 97 # a
    elif input == 32: #space
        return 32

color = np.random.randint(0, 255, (100, 3))
lk_params = dict(winSize=(21, 21),
                 #maxLevel=5,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

img_idx=0
height_tmp = 0
def optical_flow(old_frame, frame):
    global img_idx
    mask = np.zeros_like(old_frame)
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(old_gray, maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Calculate optical flow using Lucas-Kanade method
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

    # Select good points
    if p1 is not None:
        good_new = p1[st == 1]
        good_old = p0[st == 1]

    # Draw lines between the tracked points
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel().astype(int)
        c, d = old.ravel().astype(int)
        mask = cv2.line(mask, (a, b), (c, d), color[i].tolist(), 2)
        frame = cv2.circle(frame, (a, b), 5, color[i].tolist(), -1)

    # Overlay the optical flow lines on the original frame
    stacked_img = cv2.addWeighted(old_frame, 0.5 ,frame, 1-0.5,0)
    img = cv2.add(stacked_img, mask)
    img_idx+=1
    cv2.imwrite(f"./imgs/{img_idx}.png", img)


def color_detect(img,color_name):
    #default values: 160 x 120
    # The blue range will be different under different lighting conditions and can be adjusted flexibly.  H: chroma, S: saturation v: lightness
    resize_img = cv2.resize(img, (320,240), interpolation=cv2.INTER_LINEAR)  # In order to reduce the amount of calculation, the size of the picture is reduced to (160,120)
    hsv = cv2.cvtColor(resize_img, cv2.COLOR_BGR2HSV)              # Convert from BGR to HSV
    color_type = color_name

    mask = cv2.inRange(hsv,np.array([min(color_dict[color_type]), 90, 90]), np.array([max(color_dict[color_type]), 255, 255]) )           # inRange():Make the ones between lower/upper white, and the rest black
    if color_type == 'red':
            mask_2 = cv2.inRange(hsv, (color_dict['red_2'][0],0,0), (color_dict['red_2'][1],255,255))
            mask = cv2.bitwise_or(mask, mask_2)

    morphologyEx_img = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_5,iterations=1)              # Perform an open operation on the image

    # Find the contour in morphologyEx_img, and the contours are arranged according to the area from small to large.
    _tuple = cv2.findContours(morphologyEx_img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    # compatible with opencv3.x and openc4.x
    if len(_tuple) == 3:
        _, contours, hierarchy = _tuple
    else:
        contours, hierarchy = _tuple

    color_area_num = len(contours) # Count the number of contours

    x_1 = None
    w_1 = None
    y_1 = None
    h_1 = None
    if color_area_num > 0:
        max_area = 0
        for i in contours:
             # Traverse all contours
            x,y,w,h = cv2.boundingRect(i)      # Decompose the contour into the coordinates of the upper left corner and the width and height of the recognition object
            if w*h > max_area:
                max_area = w*h
            else:
                continue
            # default values - 8 and 8 
            # Draw a rectangle on the image (picture, upper left corner coordinate, lower right corner coordinate, color, line width)
            if w >= 15 and h >= 15: # Because the picture is reduced to a quarter of the original size, if you want to draw a rectangle on the original picture to circle the target, you have to multiply x, y, w, h by 4.
                x_1 = x *1# 2
                y_1 = y *1# 2
                w_1 = w *1# 2
                h_1 = h *1# 2
                cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)  # Draw a rectangular frame
                cv2.putText(img,color_type,(x,y), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,255),2)# Add character description

    return img,mask,morphologyEx_img , (x_1,w_1) , (y_1,h_1)

#object detecion and following functions

def detect_object(camera, color):
        images = []
        for i in range(3): #first approach to stabilizing the image - avarge over a few images
            img = camera.capture_array() #frame.array
            np_frame = np.array(img)
            # print(np_frame.shape)
            images.append(np_frame)

        img = np.mean(images, axis = 0)

        img = np.array(img, dtype='uint8')
        img,img_2,img_3, object_x, object_y =  color_detect(img,color)  # Color detection function
        if object_x[0] is None: #nothing found
            return False, img, img_2, img_3

        
        frame_size_x = 320

        middle = frame_size_x / 2

        if object_x[0] is None: #If the object is not yet found - continue the turn
            print('Nothing found')
            return False, img, img_2, img_3

        x = object_x[0]
        w = object_x[1]
        y = object_y[0]
        h = object_y[1]
        global height_tmp 
        height_tmp = h
        object_mid = (2*x+w)/2
        threshold = 60 #subject of change
        height_thresh = 10
        if np.abs(middle - object_mid) < threshold and h > height_thresh: #the object rectangle crosses the middle of the frame
            print('object found')
            return True, img, img_2, img_3 #object found in a good position

        return False, img, img_2, img_3

def correction_object(camera, color):
        images = []
        for i in range(3): #first approach to stabilizing the image - avarge over a few images
            img = camera.capture_array() #frame.array
            np_frame = np.array(img)
            images.append(np_frame)

        img = np.mean(images, axis = 0)

        img = np.array(img, dtype='uint8')
        img,img_2,img_3, object_x, object_y =  color_detect(img,color)  # Color detection function
        if object_x[0] is None: #nothing found
            print('Error 1 - No object Found')
            return False, img, img_2, img_3, None

        
        frame_size_x = 320

        middle = frame_size_x / 2
        direction = None

        if object_x[0] is None: #If the object is not yet found - continue the turn
            print('Error 2 - object not found')
            return True, img, img_2, img_3, None
        
        x = object_x[0]
        w = object_x[1]
        y = object_y[0]
        h = object_y[1]
        object_mid = (2*x+w)/2
        threshold = 25 #subject of change

        min_area = 45
        print(f'object middle: {object_mid:.2f}')
        print(f'Frame middle: {middle:.2f}')
        if np.abs(middle - object_mid) < threshold and h*w >= min_area: #the object rectangle crosses the middle of the frame
            print('object in the middle - moving forward')
            print('Object middle: ' + str(object_mid))
            print('Frame middle: ' + str(middle))
            return True, img, img_2, img_3, None #object found in a good position
        
        if object_mid < middle and h*w >= min_area:
            direction = 'left'
            print('Performing left')
            return False, img, img_2, img_3, direction
        elif object_mid > middle and h*w >= min_area:
            print('performing right')
            direction = 'right'
            return False, img, img_2, img_3, direction
        print('No direction set')
        return False, img, img_2, img_3, direction

def following_object(camera,color):

    size_y = 240

    images = []
    for i in range(1): #first approach to stabilizing the image - avarge over a few images
            img = camera.capture_array() #frame.array
            np_frame = np.array(img)
            images.append(np_frame)
    img = np.mean(images, axis = 0)
    img = np.array(img, dtype='uint8')
    img,img_2,img_3, object_x, object_y =  color_detect(img,color)  # Color detection function

    y = object_y[0]
    h = object_y[1]
    if y is None:
        return True , img, img_2, img_3

    threshold = 0.72 # possible threshold
    height_tresh = 10
    if y > size_y * threshold and h > height_tresh: # follow untill the object is under 1/2 of the frame
        # print('space')
        print('Object touched')
        return True, img, img_2, img_3
    return False, img, img_2, img_3


#object_detection_bool = False

def run(object_detection_bool):
    inputList = []
    with Picamera2() as camera:
        camera.preview_configuration.main.size = (320,240)
        camera.preview_configuration.main.format = "RGB888"
        camera.preview_configuration.align()
        camera.configure("preview")
        camera.start()

        while True:
            img = camera.capture_array() #frame.array

            cv2.imshow("video", img)    # OpenCV image show
            k = cv2.waitKey(1) & 0xFF
            # 27 is the ESC key, which means that if you press the ESC key to exit
            if k in[119,97,115,100,32]: # w
                img = performInput(k, camera)
                inputList.append([k, time.time(), img])
            elif k == 27 or k == 113:
                #perform the object detection and add it all into list
                #global object_detection_bool
                if object_detection_bool: 
                    object_found = False
                    while not object_found:
                        k = cv2.waitKey(1)
                        # Turning left
                        # fc.turn_left(power_val)  #Original inplementation
                        key = 97 # A - Turning left
                        img = performInput(key,camera)
                        inputList.append([key, time.time(), img])

                        time.sleep(0.17) # changing this value will increase the turn radius before detecion
                    
                        # fc.stop() Original inplementation
                    
                        key = 32 # Space - Stop
                        img = performInput(key,camera)
                        inputList.append([key, time.time(), img])

                        time.sleep(0.2) # Waiting time before taking pictures
                        object_found, img, img_2, img_3 = detect_object(camera,'red')
                        cv2.imshow("video", img)    # OpenCV image show
                        cv2.imshow("mask", img_2)    # OpenCV image show
                        cv2.imshow("morphologyEx_img", img_3)    # OpenCV image show

                    time_needed = (50/height_tmp) * 0.9

                # object found - we are out of the loop now
                # fc.forward(power_val) - Original
                    key = 119 # W - Forward
                    img = performInput(key,camera)
                    inputList.append([key, time.time(), img])
                
                    time.sleep(time_needed)
                #Starting the correction 
                    print('Starting correction')
                    correction = False
                    key = 32 # Space - Stop
                    img = performInput(key,camera)
                    inputList.append([key, time.time(), img])
                    while not correction:
                        k = cv2.waitKey(1)
                        time.sleep(0.2) # Waiting time before taking pictures
                        correction, img, img_2, img_3,direction = correction_object(camera,'red')
                        cv2.imshow("video", img)    # OpenCV image show
                        cv2.imshow("mask", img_2)    # OpenCV image show
                        cv2.imshow("morphologyEx_img", img_3)    # OpenCV image show
                    
                    
                        if correction:
                            print('object in middle - correction ended')
                            break
                        if direction == 'left':
                            key = 97 # A - Left
                            img = performInput(key,camera)
                            inputList.append([key, time.time(), img])
                        
                        elif direction == 'right':
                            key = 100 # D - right
                            img = performInput(key,camera)
                            inputList.append([key, time.time(), img])
                        time.sleep(0.12) # changing this value will increase the turn radius before detecion

                        key = 32 # Space - stop
                        img = performInput(key,camera)
                        inputList.append([key, time.time(), img])
                    
                    
                #Correction ended
                
                # fc.forward(power_val)
                    key = 119 # W - forward
                    img = performInput(key,camera)
                    inputList.append([key, time.time(), img])
                
                    object_touch = False
                    print('Waiting for the touch')
                    while not object_touch:
                        k = cv2.waitKey(1)
                        object_touch, img, img_2, img_3  = following_object(camera,'red')
                        cv2.imshow("video", img)    # OpenCV image show
                        cv2.imshow("mask", img_2)    # OpenCV image show
                        cv2.imshow("morphologyEx_img", img_3)    # OpenCV image show

                    print('Object Touched! - Performing the return') # stop - potentially go back then
                
                
                    key  = 32 # Space  - stop
                    img = performInput(key,camera)
                    inputList.append([key, time.time(), img])
              
                #reverse
                img = camera.capture_array()
                img = performInput(32,camera)
                inputList.append([32, time.time(), img])
                sleep(0.1)
                inputList.append([k, time.time(), img])
                
                prevInput = 0
                for i in reversed(inputList):
                    if prevInput == 0:
                        prevInput = i[1]
                        prev_img = i[2]
                    else:
                        multiply = 1
                        output = getOpposite(i[0])
                        duration = prevInput - i[1]
                        prevInput = i[1]

                        fc.stop()
                        sleep(0.1)
                        duration -= 0.1
                        img = camera.capture_array()

                        if output == 97 or output == 100:
                            optical_flow(prev_img,img)
                        prev_img = i[2]

                        if output == 115:
                            multiply = 1.02
                        elif output == 100:
                            multiply = 1/1.025
                        elif output == 97:
                            multiply = 1.025
                        elif output == 119:
                            multiply = 1

                        performInput(output)
                        sleep(duration*multiply)
                performInput(32)
                break

        print('quit ...')
        cv2.destroyAllWindows()
        camera.close()
