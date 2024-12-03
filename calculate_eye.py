import cv2
import mediapipe as mp
import time

# 初始化 MediaPipe 手势识别模块
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


# 定义手势规则识别函数
# 1：食指伸出，其它手指卷曲
# 2：食指和中指伸出，其它手指卷曲
# 3：食指、中指和无名指伸出，其它手指卷曲
# 4：大拇指卷曲，其它手指伸出
# 5：全部手指伸出
# 6：大拇指和小拇指伸出，其它手指卷曲
# 7：中指、食指、大拇指伸出，其它手指卷曲
# 8：食指和大拇指伸出，其它手指卷曲
# 9：小拇指伸出，其它手指卷曲
# 0：全部手指卷曲
# 指示1：大拇指伸出，其它手指卷曲
# 指示2：大拇指、食指和小拇指伸出，其它手指卷曲
# 其它：上述情况全部不符合
def recognize_gesture(hand_landmarks):
    """
    根据手部关键点匹配手势规则
    """
    # 获取所有关键点的坐标
    landmarks = hand_landmarks.landmark

    # 判断手指状态
    # 大拇指特殊：TIP 与 MCP 的相对位置
    thumb = landmarks[mp_hands.HandLandmark.THUMB_TIP].x > landmarks[mp_hands.HandLandmark.THUMB_MCP].x

    # 其余手指：TIP 与 PIP 的相对位置（y 轴，竖起时 TIP 更靠近屏幕顶部）
    index = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP].y < landmarks[mp_hands.HandLandmark.INDEX_FINGER_PIP].y
    middle = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y < landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y
    ring = landmarks[mp_hands.HandLandmark.RING_FINGER_TIP].y < landmarks[mp_hands.HandLandmark.RING_FINGER_PIP].y
    pinky = landmarks[mp_hands.HandLandmark.PINKY_TIP].y < landmarks[mp_hands.HandLandmark.PINKY_PIP].y

    # 手指状态数组：True 表示竖起，False 表示卷曲
    fingers = [thumb, index, middle, ring, pinky]

    # 匹配规则
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
        return "指示1"
    elif fingers == [True, True, False, False, True]:
        return "指示2"
    else:
        return "其它"


# 状态机逻辑
class StateMachine:
    def __init__(self):
        self.state = 1  # 初始状态
        self.digit1 = None
        self.digit2 = None
        self.processing_start_time = None  # 记录进入处理状态的时间
        self.processing_duration = 2  # 处理状态持续时间（秒）

    def update_state(self):
        """
        根据当前时间和状态动态更新状态
        """
        if self.state == 4 and self.processing_start_time is not None:
            elapsed_time = time.time() - self.processing_start_time
            if elapsed_time >= self.processing_duration:
                # 根据当前状态切换到下一个状态
                if self.digit1 is None:
                    self.state = 2
                elif self.digit2 is None:
                    self.state = 3
                else:
                    self.state = 1  # 重置为初始状态
                self.processing_start_time = None  # 清除处理时间戳

    def handle_gesture(self, gesture):
        """
        根据当前状态和手势执行逻辑
        """
        self.update_state()  # 首先检查是否需要更新状态

        if self.state == 1 and gesture == "指示1":
            self.state = 4
            self.processing_start_time = time.time()  # 记录处理开始时间
            print("进入处理状态，稍后进入待机状态2")

        elif self.state == 2 and gesture.isdigit():
            self.digit1 = int(gesture)
            print(f"记录第一个数字：{self.digit1}")
            self.state = 4
            self.processing_start_time = time.time()  # 记录处理开始时间

        elif self.state == 3 and gesture.isdigit():
            self.digit2 = int(gesture)
            print(f"记录第二个数字：{self.digit2}")
            print(f"计算结果：{self.digit1} * {self.digit2} = {self.digit1 * self.digit2}")
            self.state = 4
            self.processing_start_time = time.time()  # 记录处理开始时间


# 打开摄像头
cap = cv2.VideoCapture(0)

# 初始化状态机
state_machine = StateMachine()

# 配置手势识别
with mp_hands.Hands(
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("无法获取摄像头画面")
            break

        # 翻转图像以更自然的呈现
        frame = cv2.flip(frame, 1)
        # 转为RGB（MediaPipe需要RGB格式）
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 检测手势
        results = hands.process(rgb_frame)

        # 在画面上绘制手势检测结果
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # 绘制手部关键点和连接
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # 识别手势
                gesture = recognize_gesture(hand_landmarks)
                cv2.putText(frame, f"Gesture: {gesture}", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # 传递手势到状态机
                state_machine.handle_gesture(gesture)

        # 显示结果画面
        cv2.imshow('Hand Gesture Recognition', frame)

        # 按键退出
        if cv2.waitKey(5) & 0xFF == 27:  # 按ESC退出
            break

# 释放资源
cap.release()
cv2.destroyAllWindows()
