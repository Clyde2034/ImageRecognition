import cv2
import keras
from cv2 import face
import numpy as np
import mediapipe as mp

"""
Show Image
"""
# input1 = cv2.imread("input1.jpg")
# cv2.imshow("Windows", input1)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

"""
Save Image
"""
# image=np.zeros((500,500,3),dtype="uint8")
# image[150:350,150:350]=[0,0,255]
# cv2.imwrite("redblock.jpg",image)
# cv2.imshow("Windows",image)
# cv2.waitKey(0)

"""
Read Video
"""
# camera = cv2.VideoCapture(0)
# if camera.isOpened():
#     print("Opened")
# else:
#     print("Closed")
#     exit()
#
# while True:
#     ret, frame = camera.read()
#     if not ret:
#         print("Can not receive frame")
#         break
#     gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
#     cv2.imshow("Camera", gray)
#     if cv2.waitKey(1) == ord('q'):
#         break
# camera.release()
# cv2.destroyAllWindows()

"""
Write Video
"""
# capture = cv2.VideoCapture(0)
# width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
# height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
# fourcc = cv2.VideoWriter_fourcc(*"mp4v")
# output = cv2.VideoWriter("output01.mp4", fourcc, 20.0, (width, height),isColor=False)
# if not capture.isOpened():
#     print("Can not open camera")
#     exit()
# while True:
#     ret, frame = capture.read()
#     if not ret:
#         print("Can not receive frame")
#     gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
#     output.write(gray)
#     cv2.imshow("Windows",gray)
#     if cv2.waitKey(1)==ord('q'):
#         break
# capture.release()
# output.release()
# cv2.destroyAllWindows()

"""
輸出影像資訊
"""
# input1 = cv2.imread("input1.jpg")
# print(input1.shape)
# print(input1.size)
# print(input1.dtype)
# print(input1)
# cv2.imshow("Windows",input1)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

"""
Set bit
"""
# input1_blue = cv2.imread("input1.jpg")
# input1_blue[:, :, 1] = 0
# input1_blue[:, :, 2] = 0
# input1_green = cv2.imread("input1.jpg")
# input1_green[:, :, 0] = 0
# input1_green[:, :, 2] = 0
# input1_red = cv2.imread("input1.jpg")
# input1_red[:, :, 0] = 0
# input1_red[:, :, 1] = 0
# cv2.imshow("Window", input1_blue)
# cv2.imshow("Window", input1_green)
# cv2.imshow("Window", input1_red)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

"""
Thresh set
"""
# image = cv2.imread("input2.png")
# img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# ret1, op1 = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)
# ret2, op2 = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY_INV)
# ret3, op3 = cv2.threshold(img_gray, 127, 255, cv2.THRESH_TRUNC)
# cv2.imshow("Window", op3)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# image = cv2.imread("input1.jpg")
# image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# ret, op1 = cv2.threshold(image_gray, 127, 255, cv2.THRESH_BINARY)
# op2 = cv2.adaptiveThreshold(image_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
# op3 = cv2.adaptiveThreshold(image_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
# cv2.imshow("Windows", op3)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# def empty(iamge):
#     pass
#
# cv2.namedWindow("WindowsTackedBar")
# cv2.resizeWindow("WindowsTackedBar", 640, 320)
# cv2.createTrackbar("Hue Min", "WindowsTackedBar", 0, 179, empty)
# cv2.createTrackbar("Hue Max", "WindowsTackedBar", 179, 179, empty)
# cv2.createTrackbar("Sat Min", "WindowsTackedBar", 0, 255, empty)
# cv2.createTrackbar("Sat Max", "WindowsTackedBar", 255, 255, empty)
# cv2.createTrackbar("Val Min", "WindowsTackedBar", 0, 255, empty)
# cv2.createTrackbar("Val Max", "WindowsTackedBar", 255, 255, empty)
#
# image = cv2.imread("XiWinnie.jpg")
# image = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)
# image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
#
# while True:
#     hue_min = cv2.getTrackbarPos("Hue Min", "WindowsTackedBar")
#     hue_max = cv2.getTrackbarPos("Hue Max", "WindowsTackedBar")
#     sat_min = cv2.getTrackbarPos("Sat Min", "WindowsTackedBar")
#     sat_max = cv2.getTrackbarPos("Sat Max", "WindowsTackedBar")
#     val_min = cv2.getTrackbarPos("Val Min", "WindowsTackedBar")
#     val_max = cv2.getTrackbarPos("Val Max", "WindowsTackedBar")
#     print(hue_min, hue_max, sat_min, sat_max, val_min, val_max)
#
#     min = np.array([hue_min, sat_min, val_min])
#     max = np.array([hue_max, sat_max, val_max])
#     mask=cv2.inRange(image_hsv,min,max)
#
#     result= cv2.bitwise_and(image,image,mask=mask)
#
#     cv2.imshow("Windows", image)
#     cv2.imshow("WindowsHSV", image_hsv)
#     cv2.imshow("WindowsMask", mask)
#     cv2.imshow("WindowsResult", result)
#     if cv2.waitKey(1) == ord('q'):
#         cv2.destroyAllWindows()
#
# # 10 40 156 255 160 255

"""
圖形辨識
"""
# image = cv2.imread("shape.jpg")
# haveContours = image.copy()
# image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# # 邊緣檢測前先利用高斯模糊預處理
# blur = cv2.GaussianBlur(image, (3, 3), 2)
# # 邊緣檢測
# canny = cv2.Canny(image, 150, 200)
# # 找到輪廓
# contours, hierarchy = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
#
# for con in contours:
#     # 劃出輪廓
#     cv2.drawContours(haveContours, con, -1, (255, 0, 0), 2)
#     # 計算輪廓面積
#     area = cv2.contourArea(con)
#     if area > 500:
#         # 計算輪廓邊長
#         perimeter = cv2.arcLength(con, True)
#         # 得到輪廓頂點個數
#         vertice = cv2.approxPolyDP(con, perimeter * 0.02, True)
#         # 劃出外框
#         x, y, w, h = cv2.boundingRect(vertice)
#         cv2.rectangle(haveContours, (x, y), (x + w, y + h), (0, 255, 0), 2)
#         if len(vertice) == 3:
#             cv2.putText(haveContours, "Triangle", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
#         elif len(vertice) == 4:
#             cv2.putText(haveContours, "Rectangle", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
#         elif len(vertice) == 5:
#             cv2.putText(haveContours, "Pentagon", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
#         else:
#             cv2.putText(haveContours, "Circle", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
#
# cv2.imshow("WindowBlur", blur)
# cv2.imshow("WindowCanny", canny)
# cv2.imshow("WindowContours", haveContours)
#
# cv2.waitKey(0)
# cv2.destroyAllWindows()

"""
鏡頭人臉偵測
"""
# capture = cv2.VideoCapture(0)
# if not capture.isOpened():
#     print("Camera opend failed.")
#     exit()
# while True:
#     ret, frame = capture.read()
#     if not ret:
#         print("Camera read failed.")
#         exit()
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     # 載入模型
#     faceCascade = cv2.CascadeClassifier("face_detect.xml")
#     # 模型辨識結果
#     faceRect = faceCascade.detectMultiScale(gray, 1.1, 4)
#
#     # 劃出人臉辨識的外框
#     for (x, y, w, h) in faceRect:
#         cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
#     cv2.imshow("Camera", frame)
#     if cv2.waitKey(1) == ord('q'):
#         exit()
# capture.release()
# cv2.destroyAllWindows()

"""
蔡英文人臉訓練
"""
# cascade = cv2.CascadeClassifier("face_detect.xml")
# recognize = cv2.face.LBPHFaceRecognizer_create()
# faces = []
# ids = []
#
# for id in range(1, 21):
#     trainImage = cv2.imread(f"./Tsai-Ing-wen/{id}.jpg")
#     trainImage2Gray = cv2.cvtColor(trainImage, cv2.COLOR_BGR2GRAY)
#     image_np = np.array(trainImage2Gray, "uint8")
#     face = cascade.detectMultiScale(trainImage2Gray)
#     for (x, y, w, h) in face:
#         faces.append(image_np[y:y + h, x:x + w])
#         ids.append(1)
#
# print("Training...")
# recognize.train(faces,np.array(ids))
# recognize.save("face.xml")
# print("Training Finished...")

"""
蔡英文人臉辨識
"""
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("face.xml")
cascade = cv2.CascadeClassifier("face_detect.xml")
camera = cv2.VideoCapture(0)
if not camera.isOpened():
    print("Camera opend failed.")
    exit()
while True:
    ret, frame = camera.read()
    if not ret:
        print("Camera read failed.")
        exit()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faceRect = cascade.detectMultiScale(gray, 1.1, 4)
    names = {
        1: "Tsai-ing-wen"
    }

    for (x, y, w, h) in faceRect:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
        idnum, confidence = recognizer.predict(gray[y:y + h, x:x + w])
        print(confidence)
        if confidence < 65:
            name = names[idnum]
        else:
            name = "Can not organize."

        cv2.putText(frame, name, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Face Recognize",frame)
    if cv2.waitKey(1)==ord('q'):
        break

camera.release()
cv2.destroyAllWindows()

"""
手部辨識
"""
# camera = cv2.VideoCapture(0)
# mpHands = mp.solutions.hands
# mpHandsDrawing = mp.solutions.drawing_utils
# mpHandsDrawingStyle = mp.solutions.drawing_styles
# hands = mpHands.Hands()
#
# if not camera.isOpened():
#     print("Cant not opend camera.")
# while True:
#     ret, frame = camera.read()
#     if not ret:
#         print("Cant not catched camera.")
#         break
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     result = hands.process(gray)
#     if result.multi_hand_landmarks:
#         for hand in result.multi_hand_landmarks:
#             mpHandsDrawing.draw_landmarks(frame, hand, mpHands.HAND_CONNECTIONS
#                                           , mpHandsDrawingStyle.get_default_hand_landmarks_style()
#                                           , mpHandsDrawingStyle.get_default_hand_connections_style())
#
#     cv2.imshow("Windows", frame)
#     if cv2.waitKey(1) == ord('q'):
#         break
# cv2.destroyAllWindows()


from keras.datasets import mnist
from keras import models
from keras import layers
from keras.utils import to_categorical

# (train_image, train_label), (test_image, test_label) = mnist.load_data()
#
# network = models.Sequential()
# network.add(layers.Dense(512, activation="relu", input_shape=(28 * 28,)))
# network.add(layers.Dense(10, activation="softmax"))
# network.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"])
# fix_train_image = train_image.reshape((60000, 28 * 28)).astype('float32') / 255
# fix_test_image = test_image.reshape((10000, 28 * 28)).astype('float32') / 255
# fix_train_label = to_categorical(train_label)
# fix_test_label = to_categorical(test_label)
# result = network.fit(fix_train_image, fix_train_label, epochs=20, batch_size=128,
#                      validation_data=(fix_test_image, fix_test_label))
# test_loss, test_acc = network.evaluate(fix_test_image, fix_test_label)
# network.save("mnist.xml")