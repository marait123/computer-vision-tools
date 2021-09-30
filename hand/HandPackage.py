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
        print("handness", results.multi_handedness)
        annotated_image = cv2.flip(img.copy(), 1)

        if(not results.multi_handedness):
            return annotated_image
        image_hight, image_width, _ = img.shape
        for hand_landmarks in results.multi_hand_landmarks:
            # Print index finger tip coordinates.
            print(
                f'Index finger tip coordinate: (',
                f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width}, '
                f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_hight})'
            )
            mp_drawing.draw_landmarks(
                annotated_image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

        return annotated_image


cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()
hm = HandsModule(True)


req = urllib.request.urlopen(
    'http://192.168.1.2:4747/video?640x480')
req_res = req.read()
print("req rest", req_res)
arr = np.asarray(bytearray(), dtype=np.uint8)
img = cv2.imdecode(arr, -1)  # 'Load it as it is'

# img = cv2.imread(
#     "https://upload.wikimedia.org/wikipedia/commons/thumb/b/b6/Image_created_with_a_mobile_phone.png/440px-Image_created_with_a_mobile_phone.png")

# img = hm.plotHands(img)
print("image", img)
cv2.imshow("koko", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
# curr_time = 0
# prev_time = 0
# while True:
#     # Capture frame-by-frame
#     ret, frame = cap.read()
#     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     new_frame = hm.plotHands(frame)
#     # if frame is read correctly ret is True
#     if not ret:
#         print("Can't receive frame (stream end?). Exiting ...")
#         break
#     # Our operations on the frame come here
#     # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     # Display the resulting frame
#     curr_time = time.time()
#     fps = 1/(curr_time-prev_time)
#     prev_time = curr_time
#     frame = cv2.putText(frame, str(int(fps)), (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
#                         1, (255, 0, 255), 3, cv2.LINE_AA)

#     cv2.imshow('new_frame', new_frame)
#     if cv2.waitKey(1) == ord('q'):
#         break
