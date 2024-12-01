import mediapipe as mp
import cv2
from utils import FPS, BOUNDING_SIDE

# Camera utils
fps = FPS.fps()
bbox = BOUNDING_SIDE.bbox()

# Mediapipe Tasks
BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult
VisionRunningMode = mp.tasks.vision.RunningMode

# Define label and accuracy
r_label, r_acc = None, 0.0
l_label, l_acc = None, 0.0
label, acc = None, 0.0
index = None

# Camera asset
video = cv2.VideoCapture(0)
video.set(cv2.CAP_PROP_FRAME_WIDTH, 600)
video.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
video.set(cv2.CAP_PROP_FPS, 60)

def print_result(result: GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int):
    """Prints the gesture category name and score."""
    global r_label, r_acc, l_label, l_acc, index
    for gesture, handedness in zip(result.gestures, result.handedness):
        if gesture[0].category_name != '' and gesture[0].score > 0.5:
            print(f'{gesture[0].category_name}, {gesture[0].score:.3f}')

            index = handedness[0].category_name
            if index == 'Left':
                r_label, r_acc = gesture[0].category_name, gesture[0].score
            elif index == 'Right':
                l_label, l_acc = gesture[0].category_name, gesture[0].score
    return

options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path='Static_IMG_Processing/gesture_recognizer.task'),
    running_mode=VisionRunningMode.LIVE_STREAM,
    num_hands = 2,
    result_callback=print_result)

timestamp = 0
recognizer = GestureRecognizer.create_from_options(options)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(model_complexity=0,
        min_detection_confidence=0.5,
        max_num_hands = 2,
        min_tracking_confidence=0.5)
#mp_drawing = mp.solutions.drawing_utils
#mp_drawing_styles = mp.solutions.drawing_styles

while video.isOpened(): 
        ret, frame = video.read()
        if not ret:
            print("Ignoring empty frame")
            break
        frame = cv2.flip(frame, 1)
        frame.flags.writeable=False
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
        timestamp += 1
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        recognizer.recognize_async(mp_image, timestamp)
        
        #frame.flags.writeable=True
        #frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        bb = hands.process(frame)
        if bb.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(bb.multi_hand_landmarks, bb.multi_handedness):
                pos = handedness.classification[0].label
                #mp_drawing.draw_landmarks(
                #    frame,
                #    hand_landmarks,
                #    mp_hands.HAND_CONNECTIONS,
                #    mp_drawing_styles.get_default_hand_landmarks_style(),
                #    mp_drawing_styles.get_default_hand_connections_style())      

                if r_label or l_label:           
                    if pos == 'Right':
                        label, acc = r_label, r_acc
                    if pos == 'Left':
                        label, acc = l_label, l_acc
                    
                    bbox.bbox_draw(img=frame, hand_landmarks=hand_landmarks) 
                    bbox.bbox_show(img=frame, label=f'{pos}: {label} ({acc:.2f})')
                    
            fps.fpsCal()
            fps.FPS_FRONT_CAM_SHOW(img=frame)
            cv2.imshow("ASL Translator", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                exit()
video.release()
cv2.destroyAllWindows()