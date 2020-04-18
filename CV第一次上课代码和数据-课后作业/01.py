import matplotlib.pyplot as plt
import cv2
import numpy as np
import random

'''first part: reading the figure from the file: p.s: opencv format is BGR, and matplotlib  adnd pillow format are RGB'''

# Reading images by using cv.imread()
img_ori = cv2.imread("beauty.jpeg", 1)
print(img_ori)
print(img_ori.shape)

# use the cv2.imshow() to show the figure
cv2.imshow("beauty_girl", img_ori)  # 不要用img_ori去赋值于CV2.imshow()函数，会导致下面的函数产生dtyoe error
key = cv2.waitKey(0)
if key == 27:
    cv2.destroyAllWindows()

# plt也可以打印图片，但要注意是否为灰度图，以及通道顺序
img_ori_gray = cv2.imread("beauty.jpeg", 0)  # 通道必须改成灰色通道，使它与下方的“cmap=‘grap’”相匹配
plt.figure(figsize=(491, 350))
plt.imshow(img_ori_gray, cmap="gray")
plt.show()

# subplot子图的使用
plt.subplot(121)
plt.imshow(img_ori)
plt.subplot(122)
plt.imshow(cv2.cvtColor(img_ori, cv2.COLOR_BGR2RGB))  # plt.imshow()的函数中不需要再添加img_ori，不要画蛇添足
plt.show()


# 自己构造一个函数
def my_show(img, size=(5, 5)):
    plt.figure(figsize=size)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()


'''Second part: Imagination crop'''

my_show(img_ori[400:600, 500:700])  # 不要用img_ori[400:600][500,700], 会报错
my_show(img_ori[100:500][200:600])  # 这就不会报错  #问题一

'''Third part: Channel Split'''

# 图像的通道分割处理
B, G, R = cv2.split(img_ori)
cv2.imshow("B", B)
cv2.imshow("G", G)
cv2.imshow("R", R)
key = cv2.waitKey(0)
if key == 27:
    cv2.destroyAllWindows()


# 对不同通道的函数的进行映射（这里的函数功能是为了讲图片冷色化）
def img_cooler(img, b_increase, r_decrease):
    B, G, R = cv2.split(img)
    b_lim = 255 - b_increase
    B[B > b_lim] = 255
    B[B <= b_lim] = (b_increase + B[B <= b_lim]).astype(img.dtype)  # 使得B的type和img的type保持一致
    r_lim = r_decrease
    R[R < r_lim] = 0
    R[R > r_lim] = (R[R > r_lim] - r_decrease).astype(img.dtype)  # 使得B的type和img的type保持一致
    return cv2.merge((B, G, R))


cooler_img = img_cooler(img_ori, 30, 10)
my_show(cooler_img)


# Gramma Change
def adjust_gamma(img, gamma=1.0):
    invGamma = 1.0 / gamma
    table = []
    for i in range(256):
        table.append(((i / 255.0) ** invGamma) * 255)  # 将各通道的像素灰度值归一化处理，在利用幂指数来进行变换
    table = np.array(table).astype("uint8")  # 记得不要缩进
    return cv2.LUT(img, table)  # LUT的作用为相当于转化器，将图片输入大一编辑画的模型中，
    # 用LUT就能一次性全部转化成想要的像素值或者说图片


img_dark = cv2.imread("dark.jpg", 1)

my_show(img_dark, size=(5, 5))

img_brighter = adjust_gamma(img_dark, gamma=2)

my_show(img_brighter)

# 直方图均衡
plt.subplot(121)
plt.hist(img_dark.flatten(), 256, [0, 256], color="r")  # plt.hist()只能输出一维数组的格式，而图片确实二维或者说三维数组，
# 所以需要拉平成一维数组
plt.subplot(122)
plt.hist(img_brighter.flatten(), 256, [0, 256], color="b")
plt.show()

# YUV 色彩空间的Y 进行直方图均衡 来调亮图片
img_yuv = cv2.cvtColor(img_dark, cv2.COLOR_BGR2YUV)  # BGR 转成YUV格式， 注意 Y是在彩色图片格式中的第一位通道，指的是0
img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])  # 只对Y做直方图均衡
img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)  # 在把YUV的格式转化成BGR格式，因为my_show函数只能正确的接受BGR格式

my_show(img_output)

plt.subplot(131)
plt.hist(img_dark.flatten(), 256, [0, 256], color="r")
plt.subplot(132)
plt.hist(img_brighter.flatten(), 256, [0, 256], color="g")
plt.subplot(133)
plt.hist(img_output.flatten(), 256, [0, 256], color="b")
plt.show()

'''transform'''

## perspective transform

pts1 = np.float32([[0, 0], [0, 500], [500, 0], [500, 500]])
pts2 = np.float32([[100, 200], [340, 350], [170, 360], [140, 200]])
M = cv2.getPerspectiveTransform(pts1, pts2)
print(M)  # M值需要浮点数进行计算，所以pts1,pts2都需要转化为浮点数
img_warp = cv2.warpPerspective(img_ori, M, (500, 500))
my_show(img_warp)

#  Rotation
M = cv2.getRotationMatrix2D((img_ori.shape[1] / 2, img_ori.shape[0] / 2), 30, 1)  # 分别为图像的中心点，旋转的角度（逆时针为正）
# 和缩放比例的值，然后通过这个函数来生成M的值, 因为图片的第一个元素指的是Height,第二个元素指的是Width，所以中心得坐标应该反着过来算
img_rotate = cv2.warpAffine(img_ori, M, (img_ori.shape[1], img_ori.shape[0]))
# cv2.imshow("rotated image",img_rotate)
# key = cv2.waitKey(0)
# if key == 27:
#     cv2.destroyAllWindows()
my_show(img_rotate)
print(M)


# img_rotate2 = cv2.warpAffine(img_ori, M, (img_ori.shape[1],img_ori.shape[0]))
# my_show(img_rotate2)
# print(M)

# translation
def translation(img, tx=50.0, ty=40.0):
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    img_translation = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
    return img_translation


img_tran = translation(img_ori)
my_show(img_tran)
print(M)

# scale+rotation+translation = similarity transform
M = cv2.getRotationMatrix2D((img_ori.shape[1] / 2, img_ori.shape[0] / 2), 30, 0.5)
img_simi = cv2.warpAffine(img_ori, M, (img_ori.shape[1], img_ori.shape[0]))
img_simi = translation(img_simi)
my_show(img_simi)
print(M)

# Affine Transform
rows, cols, ch = img_ori.shape
pts1 = np.float32([[0, 0], [cols - 1, 0], [0, rows - 1]])
pts2 = np.float32([[cols * 0.2, rows * 0.1], [cols * 0.9, rows * 0.2], [cols * 0.1, rows * 0.9]])
M = cv2.getAffineTransform(pts1, pts2)
img_affine = cv2.warpAffine(img_ori, M, (cols, rows))
my_show(img_affine)
print(M)


# perspective transform
def random_warp(img, row, col):
    height, width, channels = img_ori.shape

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


M_warp, img_warp = random_warp(img_ori, img_ori.shape[0], img_ori.shape[1])
my_show(img_warp)

# 膨胀和腐蚀

# 膨胀
img_writing = cv2.imread("libai.png", 0)
plt.figure(figsize=(10, 8))
plt.imshow(img_writing, cmap="gray")
plt.show()

erode_writing = cv2.erode(img_writing, None, iterations=1)
plt.figure(figsize=(10, 8))
plt.imshow(img_writing, cmap="gray")
plt.show()
# 腐蚀
erode_writing = cv2.dilate(img_writing, None, iterations=1)
plt.figure(figsize=(10, 8))
plt.imshow(img_writing, cmap="gray")
plt.show()

# Optional homework
img_hsv = cv2.cvtColor(img_dark, cv2.COLOR_BGR2HSV)
img_hsv[:, :, 0] = cv2.equalizeHist(img_hsv[:, :, 0])
plt.subplot(121)
plt.hist(img_dark.flatten(), 256, [0, 256], color="r")
plt.subplot(122)
plt.hist(img_hsv.flatten(), 256, [0, 256], color="b")
plt.show()
