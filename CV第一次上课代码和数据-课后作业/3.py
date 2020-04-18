# 5.1 Please combine image crop, color shift, rotation and perspective transform together to complete a data augmentation
# script. Your code need to be completed in Python/C++ in .py or .cpp file with comments and readme file to indicate how to use.

import cv2
import random
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("beauty.jpeg", 1)
random_margin1 = 100
random_margin2 = 50


def my_show(img, size=(5, 5)):
    plt.figure(figsize=size)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()


def image_crop(img):
    w1 = random.randint(0, random_margin1)
    dw1 = random.randint(img.shape[1] - random_margin1 - 1, img.shape[1] - 1)
    h1 = random.randint(0, random_margin1)
    dh1 = random.randint(img.shape[0] - random_margin1 - 1, img.shape[0] - 1)
    img = img[w1:dw1, h1:dh1]
    my_show(img)
    return img


def color_shift(img, b, g, r):
    b = random.randint(-random_margin2, random_margin2)
    g = random.randint(-random_margin2, random_margin2)
    r = random.randint(-random_margin2, random_margin2)
    B, G, R = cv2.split(img)
    b_lim = 255 - b
    g_lim = 255 - g
    r_lim = 255 - r
    if b_lim > 255:
        b_lim = 255
    else:
        b_lim = b_lim
        B[B >= b_lim] = 255
        B[B < b_lim] = b + B[B < b_lim].astype(img.dtype)
    if g_lim > 255:
        g_lim = 255
    else:
        g_lim = g_lim
        G[G >= g_lim] = 255
        G[G < g_lim] = g + G[G < g_lim].astype(img.dtype)
    if r_lim > 255:
        r_lim = 255
    else:
        r_lim = r_lim
        R[R >= r_lim] = 255
        R[R < r_lim] = r + R[r < r_lim].astype(img.dtype)
    return cv2.merge((B, G, R))


def rotation(img):
    M = cv2.getRotationMatrix2D((img.shape[1] / 2, img.shape[0] / 2), 30, 1)  # 分别为图像的中心点，旋转的角度（逆时针为正）
    # 和缩放比例的值，然后通过这个函数来生成M的值, 因为图片的第一个元素指的是Height,第二个元素指的是Width，所以中心得坐标应该反着过来算
    img_rotate = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
    return img_rotate
    print(M)


def perspective_transform(img, row, col):  # you code here
    height, width, channels = img.shape

    # warp:
    random_margin = 60  # 随机值
    x1 = random.randint(-random_margin, random_margin)
    y1 = random.randint(-random_margin, random_margin)
    x2 = random.randint(width - random_margin - 1, width - 1)
    y2 = random.randint(-random_margin, random_margin)
    x3 = random.randint(width - random_margin - 1, width - 1)
    y3 = random.randint(height - random_margin - 1, height - 1)
    x4 = random.randint(-random_margin, random_margin)
    y4 = random.randint(height - random_margin - 1, height - 1)

    dx1 = random.randint(-random_margin, random_margin)
    dy1 = random.randint(-random_margin, random_margin)
    dx2 = random.randint(width - random_margin - 1, width - 1)
    dy2 = random.randint(-random_margin, random_margin)
    dx3 = random.randint(width - random_margin - 1, width - 1)
    dy3 = random.randint(height - random_margin - 1, height - 1)
    dx4 = random.randint(-random_margin, random_margin)
    dy4 = random.randint(height - random_margin - 1, height - 1)

    pts1 = np.float32([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
    pts2 = np.float32([[dx1, dy1], [dx2, dy2], [dx3, dy3], [dx4, dy4]])
    M_wrap = cv2.getPerspectiveTransform(pts1, pts2)
    img_warp = cv2.warpPerspective(img, M_wrap, (width, height))
    return M_wrap, img_warp


img_crop = image_crop(img)
my_show(img_crop)
img_shift = color_shift(img, 30, 40, 20)
my_show(img_shift)
img_rotate = rotation(img)
my_show(img_rotate)
M_warp, img_warp = perspective_transform(img, img.shape[0], img.shape[1])
my_show(img_warp)
