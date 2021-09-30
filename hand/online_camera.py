import numpy as np
import urllib
import cv2
import mediapipe as mp
import time


mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


class HandsModule():
    def __init__(self, mode=True, maxHands=2, detectConf=0.5, trackConf=.5):
        self.hands = mp_hands.Hands(
            static_image_mode=mode,
            max_num_hands=maxHands,
            min_detection_confidence=detectConf,
            min_tracking_confidence=trackConf)

    def plotHands(self, img):
        results = self.hands .process(
            cv2.flip(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), 1))
        # print("handness", results.multi_handedness)
        annotated_image = cv2.flip(img.copy(), 1)

        if(not results.multi_handedness):
            return annotated_image
        image_hight, image_width, _ = img.shape
        for hand_landmarks in results.multi_hand_landmarks:
            # Print index finger tip coordinates.
            # print(
            #     f'Index finger tip coordinate: (',
            #     f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width}, '
            #     f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_hight})'
            # )
            mp_drawing.draw_landmarks(
                annotated_image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

        return annotated_image

    def get_hands_fingers_pos(self, img):
        coords = []

        results = self.hands .process(
            cv2.flip(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), 1))
        print("handness", results.multi_handedness)
        annotated_image = cv2.flip(img.copy(), 1)

        if(not results.multi_handedness):
            return []
        image_hight, image_width, _ = img.shape
        for hand_landmarks in results.multi_hand_landmarks:
            # Print index finger tip coordinates.
            print(
                f'Index finger tip coordinate: (',
                f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width}, '
                f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_hight})'
            )
            one_hand_coord = [(int(l.x*image_width), int(l.y*image_hight))
                              for l in hand_landmarks.landmark]
            coords.append(one_hand_coord)
            mp_drawing.draw_landmarks(
                annotated_image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

        return coords


cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()
hm = HandsModule()


cam = "http://192.168.1.2:4747/video?640x480"
# cam = 0 # Use  local webcam.
print("before capture")
# cap = cv2.VideoCapture(cam) # for internet
cap = cv2.VideoCapture(0)
print("after capture")
if not cap:
    print("!!! Failed VideoCapture: invalid parameter!")
prev_time = 0
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
        for coord in coords:
            cv2.circle(flipped_frame, coord[8],
                       25, (255, 0, 255), 3, cv2.FILLED)
            cv2.circle(flipped_frame, coord[4],
                       25, (255, 0, 255), 3, cv2.FILLED)
            flipped_frame = cv2.putText(flipped_frame, str(coord[4]), coord[4], cv2.FONT_HERSHEY_SIMPLEX,
                                        1, (255, 0, 255), 3, cv2.LINE_AA)

            flipped_frame = cv2.putText(flipped_frame, str(coord[8]), coord[8], cv2.FONT_HERSHEY_SIMPLEX,
                                        1, (255, 0, 255), 3, cv2.LINE_AA)

    cv2.imshow('flipped_frame', flipped_frame)
    # cv2.imshow('frame', new_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
