import numpy as np
import urllib
import cv2
import mediapipe as mp
import time
# import winsound
import HandModule 
import threading
import math
import pygame

pygame.init()
pygame.mixer.init()
sound_paths=['zill1.wav','zill2.wav']
sounds = [pygame.mixer.Sound(p) for p in sound_paths]
   # Now plays at 90% of full volume.
def zill(z=0, vol=1):
    # lock.acquire()
    z=z%len(sounds)
    sounds[z].set_volume(vol)
    sounds[z].play()     
    # lock.release()

def distance(p1, p2):
    return int(math.sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2))

# zill()

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()
hm = HandModule.HandsModule()


cam = "http://192.168.1.2:4747/video?640x480"
# cam = 0 # Use  local webcam.
print("before capture")
# cap = cv2.VideoCapture(cam) # for internet
cap = cv2.VideoCapture(0)
print("after capture")
if not cap:
    print("!!! Failed VideoCapture: invalid parameter!")
prev_time = 0
hands={}
th=None
counter=0
while(True):
    # Capture frame-by-frame
    ret, current_frame = cap.read()
    # Display the resulting frame
    # new_frame = hm.plotHands(current_frame)
    # coords = []
    coords = hm.get_hands_fingers_pos(current_frame)
    # print("coords", coords)
    flipped_frame = cv2.flip(current_frame, 1)
    curr_time = time.time()
    fps = 1/(curr_time-prev_time)
    prev_time = curr_time
    flipped_frame = cv2.putText(flipped_frame, str(int(fps)), (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                                1, (255, 0, 255), 3, cv2.LINE_AA)

    if(len(coords)):
        for i, coord in enumerate(coords):
            cv2.circle(flipped_frame, coord[8],
                       10, (255, 0, 255), 2, cv2.FILLED)
            cv2.circle(flipped_frame, coord[4],
                       10, (255, 0, 255), 2, cv2.FILLED)
            # flipped_frame = cv2.putText(flipped_frame, str(coord[4]), coord[4], cv2.FONT_HERSHEY_SIMPLEX,
            #                             1, (255, 0, 255), 3, cv2.LINE_AA)

            # flipped_frame = cv2.putText(flipped_frame, str(coord[8]), coord[8], cv2.FONT_HERSHEY_SIMPLEX,
            #                             1, (255, 0, 255), 3, cv2.LINE_AA)
            dis=distance(coord[4],coord[8])
            up_limit = distance(coord[8], coord[5])//2
            down_limit = distance(coord[8], coord[6])
            if not i  in hands:
                hands[i] = {
                        'big_dist':dis > up_limit
                }
            if dis > up_limit:
                hands[i] = {
                        'big_dist':dis > up_limit
                }
            if 'big_dist' in hands[i] and hands[i]['big_dist'] and dis < down_limit:
                th=threading.Thread(target=zill,args=(i,1))
                th.start()
                print("beeb", (counter)); counter+=1
                
                hands[i] = {
                        'big_dist':False
                }
            # flipped_frame = cv2.putText(flipped_frame, f'{i}: {int(dis)}', coord[0], cv2.FONT_HERSHEY_SIMPLEX, 
            #     1, (255, 0, 255), 3, cv2.LINE_AA)
            flipped_frame = cv2.putText(flipped_frame, f'{down_limit} / {up_limit}', (100,50), cv2.FONT_HERSHEY_SIMPLEX, 
                1, (255, 0, 255), 3, cv2.LINE_AA)
            
    cv2.imshow('flipped_frame', flipped_frame)
    # cv2.imshow('frame', new_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break