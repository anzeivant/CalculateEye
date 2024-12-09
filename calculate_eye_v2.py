import cv2
import mediapipe as mp
import time
import math
import serial

# 初始化 Arduino 的串口连接
arduino_port = "COM3"  # 替换为你的 Arduino 端口
baud_rate = 9600        # 与 Arduino 中的 Serial.begin 保持一致
try:
    arduino = serial.Serial(arduino_port, baud_rate, timeout=1)
    time.sleep(2)  # 等待 Arduino 初始化
    print(f"Connected to Arduino on {arduino_port}")
except Exception as e:
    print(f"Error connecting to Arduino: {e}")
    arduino = None

# 初始化 MediaPipe 手势识别模块
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def calculate_angle_between_vectors(v1, v2):
    """
    计算两个向量之间的夹角
    """
    dot_product = v1[0] * v2[0] + v1[1] * v2[1]
    magnitude_v1 = math.sqrt(v1[0]**2 + v1[1]**2)
    magnitude_v2 = math.sqrt(v2[0]**2 + v2[1]**2)
    if magnitude_v1 == 0 or magnitude_v2 == 0:  # 避免除以零
        return 0
    angle = math.acos(dot_product / (magnitude_v1 * magnitude_v2))
    return math.degrees(angle)

def recognize_gesture(hand_landmarks, handedness):
    """
    根据手部关键点匹配手势规则
    """
    landmarks = hand_landmarks.landmark
    thumb_mcp = (landmarks[mp_hands.HandLandmark.THUMB_MCP].x, landmarks[mp_hands.HandLandmark.THUMB_MCP].y)
    thumb_ip = (landmarks[mp_hands.HandLandmark.THUMB_IP].x, landmarks[mp_hands.HandLandmark.THUMB_IP].y)
    thumb_tip = (landmarks[mp_hands.HandLandmark.THUMB_TIP].x, landmarks[mp_hands.HandLandmark.THUMB_TIP].y)

    vector1 = (thumb_ip[0] - thumb_mcp[0], thumb_ip[1] - thumb_mcp[1])
    vector2 = (thumb_tip[0] - thumb_ip[0], thumb_tip[1] - thumb_ip[1])

    thumb_angle = calculate_angle_between_vectors(vector1, vector2)
    thumb = thumb_angle < 25

    index = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP].y < landmarks[mp_hands.HandLandmark.INDEX_FINGER_PIP].y
    middle = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y < landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y
    ring = landmarks[mp_hands.HandLandmark.RING_FINGER_TIP].y < landmarks[mp_hands.HandLandmark.RING_FINGER_PIP].y
    pinky = landmarks[mp_hands.HandLandmark.PINKY_TIP].y < landmarks[mp_hands.HandLandmark.PINKY_PIP].y

    fingers = [thumb, index, middle, ring, pinky]

    if fingers == [False, True, False, False, False]:
        return "1"
    elif fingers == [False, True, True, False, False]:
        return "2"
    elif fingers == [False, True, True, True, False]:
        return "3"
    elif fingers == [False, True, True, True, True]:
        return "4"
    elif fingers == [True, True, True, True, True]:
        return "5"
    elif fingers == [True, False, False, False, True]:
        return "6"
    elif fingers == [True, True, True, False, False]:
        return "7"
    elif fingers == [True, True, False, False, False]:
        return "8"
    elif fingers == [False, False, False, False, True]:
        return "9"
    elif fingers == [False, False, False, False, False]:
        return "0"
    elif fingers == [True, False, False, False, False]:
        return "command2"
    elif fingers == [True, True, False, False, True]:
        return "command1"
    else:
        return "unknown"

class Eye:
    def __init__(self):
        self.state = 1
        self.state_time = 0
        self.last_time = 0
        self.deal_Timer = 2
        self.last_state = 1
        self.digit1 = 0
        self.digit2 = 0

    def roll_eye(self, direction):
        command = f"ROLL:{direction}\n"
        if arduino:
            arduino.write(command.encode())
            print(f"Sent to Arduino: {command.strip()}")

    def twink_eye(self, count):
        command = f"TWINK:{count}\n"
        if arduino:
            arduino.write(command.encode())
            print(f"Sent to Arduino: {command.strip()}")

    def show_number(self, number):
        second = number % 10
        first = int((number - second) / 10)

        if first > 0:
            self.roll_eye(first)
            self.twink_eye(1)

        self.roll_eye(second)
        print(f"Displayed number: {first * 10 + second}")

    def update_with_gesture(self, gesture):
        time_now = time.time()
        self.state_time += time_now - self.last_time
        self.last_time = time_now

        if self.state == 0:
            if self.state_time > self.deal_Timer:
                next_state = self.last_state + 1
                if next_state > 3:
                    next_state = 1
                self.transition_state(next_state)

        elif self.state == 1:
            if gesture == "command1":
                self.transition_state(0)
                self.twink_eye(2)

        elif self.state == 2:
            if gesture.isdigit():
                self.digit1 = int(gesture)
                print(f"Received digit: {gesture}")
                self.transition_state(0)
                self.twink_eye(2)

        elif self.state == 3:
            if gesture.isdigit():
                self.digit2 = int(gesture)
                print(f"Received digit: {gesture}")
                self.transition_state(0)
                self.twink_eye(2)
                self.show_number(self.digit1 * self.digit2)
                self.twink_eye(2)

    def transition_state(self, next_state):
        self.last_state = self.state
        self.state = next_state
        self.state_time = 0

eye = Eye()

class FixedFrameRate:
    def __init__(self, target_fps):
        self.current_frame_time = time.time()
        self.frame_interval = 1.0 / target_fps
        self.last_frame_time = time.time()

    def start_frame(self):
        self.last_frame_time = self.current_frame_time
        self.current_frame_time = time.time()

    def end_frame(self):
        elapsed_time = time.time() - self.current_frame_time
        sleep_time = self.frame_interval - elapsed_time
        if sleep_time > 0:
            time.sleep(sleep_time)

    def delta_time(self):
        return self.current_frame_time - self.last_frame_time

frame_rate = FixedFrameRate(target_fps=10)
cap = cv2.VideoCapture(0)

with mp_hands.Hands(
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        frame_rate.start_frame()

        ret, frame = cap.read()
        if not ret:
            print("无法获取摄像头画面")
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmarks, hand_handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                handedness = hand_handedness
                label = handedness.classification[0].label

                gesture = recognize_gesture(hand_landmarks, handedness)
                cv2.putText(frame, f"Gesture: {gesture}", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                eye.update_with_gesture(gesture)

        cv2.imshow('Hand Gesture Recognition', frame)

        if cv2.waitKey(5) & 0xFF == 27:
            break

        frame_rate.end_frame()

cap.release()
cv2.destroyAllWindows()
if arduino:
    arduino.close()
