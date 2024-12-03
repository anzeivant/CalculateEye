# 导入所需要的库
import cv2
import mediapipe as mp
import time
import math

# 初始化 MediaPipe 手势识别模块
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


def calculate_angle_between_vectors(v1, v2):
    """
    计算两个向量之间的夹角
    :param v1: 第一个向量 (x1, y1)
    :param v2: 第二个向量 (x2, y2)
    :return: 夹角（以度为单位）
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
    根据手部关键点匹配手势规则，改进大拇指判断
    """
    # 获取所有关键点的坐标
    landmarks = hand_landmarks.landmark

    # 获取大拇指的三个关键点坐标
    thumb_mcp = (landmarks[mp_hands.HandLandmark.THUMB_MCP].x,
                 landmarks[mp_hands.HandLandmark.THUMB_MCP].y)
    thumb_ip = (landmarks[mp_hands.HandLandmark.THUMB_IP].x,
                landmarks[mp_hands.HandLandmark.THUMB_IP].y)
    thumb_tip = (landmarks[mp_hands.HandLandmark.THUMB_TIP].x,
                 landmarks[mp_hands.HandLandmark.THUMB_TIP].y)

    # 计算两个向量：MCP->IP 和 IP->TIP
    vector1 = (thumb_ip[0] - thumb_mcp[0], thumb_ip[1] - thumb_mcp[1])
    vector2 = (thumb_tip[0] - thumb_ip[0], thumb_tip[1] - thumb_ip[1])

    # 计算夹角
    thumb_angle = calculate_angle_between_vectors(vector1, vector2)

    # 判断大拇指状态（接近 180 度为伸直，否则为弯曲）
    thumb = thumb_angle < 25

    # 其余手指状态：TIP 与 PIP 的相对位置（y 轴）
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
        return "command2"
    elif fingers == [True, True, False, False, True]:
        return "command1"
    else:
        return "unknown"


# 抽象眼球对象
class Eye:
    # 初始化
    def __init__(self):
        # 定义当前属于的状态
        # 0：处理状态
        # 1：待机状态1
        # 2：待机状态2
        # 3：待机状态3
        self.state = 1
        self.state_time = 0  # 当前状态持续的时间
        self.last_time = 0  # 记录上一次更新的时间
        self.deal_Timer = 2  # 处理状态时需要等待的时间
        self.last_state = 1  # 记录上一个状态
        self.digit1 = 0  # 记录接受到的第一个数字
        self.digit2 = 0  # 记录接受到的第二个数字

    # 转动眼球
    def roll_eye(self, direction):
        if direction == 0:
            print('眼球旋转一圈')
        elif direction == 1:
            print('眼球向左下方转动')
        elif direction == 2:
            print('眼球向下方转动')
        elif direction == 3:
            print('眼球向右下方转动')
        elif direction == 4:
            print('眼球向左方转动')
        elif direction == 5:
            print('眼球静静地看着你')
        elif direction == 6:
            print('眼球向右方转动')
        elif direction == 7:
            print('眼球向左上方转动')
        elif direction == 8:
            print('眼球向上方转动')
        elif direction == 9:
            print('眼球向右上方转动')

    # 眨眼
    def twink_eye(self, count):
        if count == 1:
            print('眼球快速地眨了一下眼')
        elif count == 2:
            print('眼球连续眨了两下眼')

    # 显示数字
    def show_number(self, number):

        second = number % 10
        first = int((number - second) / 10)

        if first > 0:
            self.roll_eye(first)
            self.twink_eye(1)

        self.roll_eye(second)

        print('眼球表示了乘法结果'+str(first*10+second))

    # 处理函数
    def update_with_gesture(self, gesture):

        # 更新状态时间
        time_now = time.time()
        self.state_time += time_now - self.last_time
        self.last_time = time_now

        # 状态逻辑处理
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
                print('眼球接受到了数字' + gesture)
                self.transition_state(0)
                self.twink_eye(2)

        elif self.state == 3:
            if gesture.isdigit():
                self.digit2 = int(gesture)
                print('眼球接受到了数字' + gesture)
                self.transition_state(0)
                self.twink_eye(2)
                self.show_number(self.digit1 * self.digit2)
                self.twink_eye(2)

    # 状态转换函数
    def transition_state(self, next_state):

        # 改变状态
        self.last_state = self.state
        self.state = next_state
        self.state_time = 0


# 获取眼球对象
eye = Eye()


# 固定帧率需要用到的类
class FixedFrameRate:
    def __init__(self, target_fps):
        self.current_frame_time = time.time()
        self.frame_interval = 1.0 / target_fps  # 每帧时间间隔
        self.last_frame_time = time.time()  # 上一帧的时间

    def start_frame(self):
        """
        在每帧逻辑的开始调用，记录帧时间
        """
        self.last_frame_time = self.current_frame_time  # 更新上一帧的时间
        self.current_frame_time = time.time()

    def end_frame(self):
        """
        在每帧逻辑的结束调用，控制帧率
        """
        elapsed_time = time.time() - self.current_frame_time
        sleep_time = self.frame_interval - elapsed_time
        if sleep_time > 0:
            time.sleep(sleep_time)  # 延时补足剩余时间

    def delta_time(self):
        """
        获取当前帧的实际运行时间（Delta Time）
        """
        return self.current_frame_time - self.last_frame_time


# 获取帧控制对象
frame_rate = FixedFrameRate(target_fps=10)  # 目标帧率 10 FPS

# 打开摄像头
cap = cv2.VideoCapture(0)

# 主要循环逻辑
with mp_hands.Hands(
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        # 帧控制处理
        frame_rate.start_frame()

        # 处理输入逻辑
        # 更新逻辑
        # 渲染逻辑
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
            for hand_landmarks, hand_handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                # 绘制手部关键点和连接
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # 获取左右手分类信息
                handedness = hand_handedness
                label = handedness.classification[0].label  # 'Left' 或 'Right'

                # 识别手势
                gesture = recognize_gesture(hand_landmarks, handedness)
                cv2.putText(frame, f"Gesture: {gesture}", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # 传递手势到状态机
                eye.update_with_gesture(gesture)

        # 显示结果画面
        cv2.imshow('Hand Gesture Recognition', frame)

        # 按键退出
        if cv2.waitKey(5) & 0xFF == 27:  # 按ESC退出
            break

        # 帧控制处理
        frame_rate.end_frame()

        # 可选：打印帧时间
        # print(f"Delta Time: {frame_rate.delta_time():.4f} seconds")

# 释放资源
cap.release()
cv2.destroyAllWindows()
