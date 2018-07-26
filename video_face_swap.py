#!/usr/bin/env python
# -*- coding: utf-8 -*-
# video_face_swap.py
import os
import numpy as np
import sys
import time
import cv2
import dlib

from keras.preprocessing import image as imagekeras
from keras.models import load_model
from PIL import Image, ImageDraw, ImageFont

size = 150
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'keras_face_trained_model.h5'


# 类别编码转换为英文名称返回
def return_name_en(codelist):
    names = ['', '', '']
    for it in range(0, len(codelist), 1):
        if int(codelist[it]) == 1.0:
            return names[it]


# 换脸
def swap_face(img1, rectangle1, rectangle2):
    # 加载新脸图片
    img2 = cv2.imread('dog2.png')
    # 获取人脸宽度
    w = (rectangle1.right() - rectangle1.left()) *1.2
    w = int(w)
    # 调整新脸图片大小
    img2 = cv2.resize(img2, (w, w))

    # 截取背景图片被替换部分图片
    top = rectangle1.top() - 20
    left = rectangle1.left() - 20
    roi = img1[top:top + w, left:left + w]

    # 生成新脸掩码
    img2gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(img2gray, 254, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)

    # 取roi 中与mask 中不为零的值对应的像素的值，其他值为0
    # 注意这里必须有mask=mask 或者mask=mask_inv, 其中的mask= 不能忽略
    img1_bg = cv2.bitwise_and(roi, roi, mask=mask)
    # 取roi 中与mask_inv 中不为零的值对应的像素的值，其他值为0。
    # 提取新脸图片中脸部分（背景去掉）
    img2_fg = cv2.bitwise_and(img2, img2, mask=mask_inv)

    # 背景和新脸合成
    dst = cv2.add(img1_bg, img2_fg)
    img1[top:top + w, left:left + w] = dst

	
	
	    # 获取人脸宽度
    w = (rectangle2.right() - rectangle2.left()) *1.2
    w = int(w)
    # 调整新脸图片大小
    img2 = cv2.resize(img2, (w, w))

    # 截取背景图片被替换部分图片
    top = rectangle2.top() - 20
    left = rectangle2.left() - 20
    roi = img1[top:top + w, left:left + w]

    # 生成新脸掩码
    img2gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(img2gray, 254, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)

    # 取roi 中与mask 中不为零的值对应的像素的值，其他值为0
    # 注意这里必须有mask=mask 或者mask=mask_inv, 其中的mask= 不能忽略
    img1_bg = cv2.bitwise_and(roi, roi, mask=mask)
    # 取roi 中与mask_inv 中不为零的值对应的像素的值，其他值为0。
    # 提取新脸图片中脸部分（背景去掉）
    img2_fg = cv2.bitwise_and(img2, img2, mask=mask_inv)

    # 背景和新脸合成
    dst = cv2.add(img1_bg, img2_fg)
    img1[top:top + w, left:left + w] = dst
	
    return img1


# 区分和标记视频中截图的人脸
def face_rec():
    global image_ouput
    model = load_model(os.path.join(save_dir, model_name))
    #camera = cv2.VideoCapture("2.mp4")  # 视频
    # camera = cv2.VideoCapture(0) # 摄像头
    
    index = 0
    while (index < 10000):
        print(index)
        try:
            time.sleep(5)
            img = cv2.imread("img/test{id}.jpg".format(id = index))
            winname = "affection"
            cv2.namedWindow(winname)
            cv2.moveWindow(winname, 1000, 100)
            cv2.imshow("affection", img)
            #cv2.waitKey()
            #try:
                # 未截取视频图片结束本次循环
                #if not (type(img) is np.ndarray):
                    #continue
                #gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 图片转为灰度图
            #except:
                #print("Unexpected error:", sys.exc_info()[0])
                #break
            gray_img = cv2.imread("img/test{id}.jpg".format(id = index), 0)
            index = (index+1)%6
            # 使用detector进行人脸检测
            # 使用dlib自带的frontal_face_detector作为我们的特征提取器
            detector = dlib.get_frontal_face_detector()
            dets = detector(gray_img, 1)  # 提取截图中所有人脸

            facelist = []
            for i, d in enumerate(dets):  # 依次区分截图中的人脸
                x1 = d.top() if d.top() > 0 else 0
                y1 = d.bottom() if d.bottom() > 0 else 0
                x2 = d.left() if d.left() > 0 else 0
                y2 = d.right() if d.right() > 0 else 0

                img = cv2.rectangle(img, (x2, x1), (y2, y1), (255, 0, 0), 2)  # 人脸画框
                face = img[x1:y1, x2:y2]
                face = cv2.resize(face, (size, size))
                x_input = np.expand_dims(face, axis=0)
                prey = model.predict(x_input)  # 人脸标记预测

                facelist.append([d, return_name_en(prey[0])])  # 存储一张图中多张人脸坐标和标记（姓名）

            cv2_im = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # cv2和PIL中颜色的hex码的储存顺序不同
            pil_im = Image.fromarray(cv2_im)
            # 图片上打印
            draw = ImageDraw.Draw(pil_im)
            # 第一个参数为字体文件路径，第二个为字体大小
            font = ImageFont.truetype("simhei.ttf", 20, encoding="utf-8")
            if len(facelist) != 2:
                continue;
            for i in facelist:
                # 人脸标记写入图片，第一个参数为打印的坐标，第二个为打印的文本，第三个为字体颜色，第四个为字体
                draw.text((i[0].left() + int((i[0].right() - i[0].left()) / 2 - len(i[1]) * 10), i[0].top() - 20), i[1],
                          (255, 0, 0), font=font)

                # PIL图片转换为cv2图片
                cv2_char_img = cv2.cvtColor(np.array(pil_im), cv2.COLOR_RGB2BGR)

                # 随机人脸用新脸替换
                #print(facelist)
                try:
                    cv2_swap_img = swap_face(cv2_char_img, facelist[0][0], facelist[1][0])
                except:
                    continue

        # 显示标记和换脸后图片
            winname = "camera"
            cv2.namedWindow(winname)
            cv2.moveWindow(winname, 500, 100)
            cv2.imshow("camera", cv2_swap_img)
            if cv2.waitKey(1) & 0xff == ord("q"):
                break

        except:
            continue
    cv2.destroyAllWindows()


if __name__ == "__main__":
    face_rec()
