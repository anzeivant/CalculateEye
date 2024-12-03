import cv2
import mediapipe as mp

# 初始化 MediaPipe 手势识别模块
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# 定义规则识别函数
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

# 打开摄像头
cap = cv2.VideoCapture(0)

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
                cv2.putText(frame, f"Gesture: {gesture}", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # 显示结果画面
        cv2.imshow('Hand Gesture Recognition', frame)

        # 按键退出
        if cv2.waitKey(5) & 0xFF == 27:  # 按ESC退出
            break

# 释放资源
cap.release()
cv2.destroyAllWindows()


# 主要逻辑
    # 定义当前处于的状态 1：待机状态1 2：待机状态2 3：待机状态3 4：正在处理状态

    # 循环执行
        # 获取摄像头画面
        # 解析当前手势
        # 进行交互（状态改变时在控制台输出当前状态
            # 如果当前是正在处理状态则不进行任何交互
            # 如果当前是待机状态1，且检测到手势“指示1”，则进入正在处理状态，一段时间后进入待机状态2
            # 如果当前是待机状态2，且检测到数字手势，则保留当前数字，记为digit1并输出控制台，然后进入正在处理状态，一段时间后进入待机状态3
            # 如果当前是待机状态3，且检测到数字手势，则保留当前数字，记为digit2并输出控制台，然后进入正在处理状态，输出digit1*digit2，一段时间后返回待机状态1

# 清理资源