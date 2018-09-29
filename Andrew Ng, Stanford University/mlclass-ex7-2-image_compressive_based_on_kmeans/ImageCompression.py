"""
Which is requested in the Exercise7 Task2 is:
    In this exercise, you will apply K-means to image compression. In a straightforward 24-bit color representation of an image,
1 each pixel is represented as three 8-bit unsigned integers (ranging from 0 to 255) that specify the red, green and blue 
intensity values. This encoding is often refered to as the RGB encoding. Our image contains thousands of colors, and in this 
part of the exercise, you will reduce the number of colors to 16 colors.
    By making this reduction, it is possible to represent (compress) the photo in an efficient way. Specifically, you only 
need to store the RGB values of the 16 selected colors, and for each pixel in the image you now need to only store the index of 
the color at that location (where only 4 bits are necessary to represent 16 possibilities).
    In this exercise, you will use the K-means algorithm to select the 16 colors that will be used to represent the compressed image. 
Concretely, you will treat every pixel in the original image as a data example and use the K-means algorithm to find the 16 colors 
that best group (cluster) the pixels in the 3-dimensional RGB space. Once you have computed the cluster centroids on the image, 
you will then use the 16 colors to replace the pixels in the original image.
"""

import numpy as np
import load_data
from KMeans import KMeans
import cv2

class ImageCompressionBasedOnData():
    '''
    根据像素的值进行聚类
    为什么欧式距离效果比余弦相似度效果好？
    '''
    def __init__(self, n_color=16):
        self.estimator = None
        self.n_color = n_color
    def fit(self, image_RGB, max_iter=3, min_move=5.0, mode='Cosine'):
        (H, W, C) = image_RGB.shape             
        image_flatten = image_RGB.reshape(H * W, C)                     # 展开图片
        self.estimator = KMeans(n_cluster=self.n_color, mode=mode)      # 进行聚类
        self.estimator.fit(image_flatten, max_iter=max_iter, min_move=min_move)    # 训练模型
    def transform(self, image_RGB):
        (H, W, C) = image_RGB.shape
        image_flatten = image_RGB.reshape((H * W, C))                   # 展开图片
        labels = self.estimator.predict(image_flatten)                  # 预测，返回各个像素点对应的簇中心索引
        image_replaced_flatten = self.estimator.centroids[labels].astype('int') # 利用索引数组进行查表，获取替换后的图片
        image_replaced = image_replaced_flatten.reshape((H, W, C))      # 重新组合为三通道图片
        return image_replaced

class ImageCompressionBasedOnLocation():
    '''
    第二种方法，吴恩达文档中的方法，根据图片像素的索引进行聚类
      This creates a three-dimensional matrix A whose first two indices identify
    a pixel position and whose last index represents red, green, or blue. For
    example, A(50, 33, 3) gives the blue intensity of the pixel at row 50 and
    column 33.
      The code inside ex7.m first loads the image, and then reshapes it to create
    an m × 3 matrix of pixel colors (where m = 16384 = 128 × 128), and calls
    your K-means function on it

    问题: 如果只采用{1, 2, 3}表示RGB，有3x3x3=27种，但怎么表示颜色的深浅程度
    暂时解决办法：在原三通道图的各个通道上进行二值化处理？
    '''
    def __init__(self, n_color=16):
        self.estimator = None
        self.n_color = n_color
    def encode(self, image_RGB):
        '''
        编码函数
        '''
        # (W, H, C) = image_RGB.shape
        # dst = image_RGB.copy()
        # for c in range(3):
        #     retval, dst[:, :, c] = cv2.threshold(image_RGB[:, :, c], thresh=255//2, maxval=c, cv2.THRESH_BINARY)
        # todo
        pass
    def decode(self, image_flatten_index):
        '''
        解码函数
        '''
        # todo
        pass
    def fit(self, image_RGB, mode='Euclidean'):
        # todo
        image_flatten_index = self.encode(image_RGB)
        pass
    def transform(self, image_RGB):
        # todo
        pass
        

if __name__ == '__main__':
    # imageData = load_data.load_image_28x28x3(load_data.image_bird, display=True)  # H x W x C

    # ------------------------------------------------------------------------------
    # IC = ImageCompressionBasedOnData()
    # IC.fit(imageData, mode='Cosine')
    # image_res = IC.transform(imageData)
    # cv2.imwrite('./data/bird_small_based_on_data_cosine.png', image_res)

    # IC.fit(imageData, mode='Euclidean')
    # image_res = IC.transform(imageData)
    # cv2.imwrite('./data/bird_small_based_on_data_euclidean.png', image_res)

    # ------------------------------------------------------------------------------
    # IC = ImageCompressionBasedOnLocation()
    # IC.fit(imageData, mode='Cosine')
    # image_res = IC.transform(imageData)
    # cv2.imwrite('./data/bird_small_based_on_location_cosine.png', image_res)

    # IC.fit(imageData, mode='Euclidean')
    # image_res = IC.transform(imageData)
    # cv2.imwrite('./data/bird_small_based_on_location_euclidean.png', image_res)


    ########################################################################################
    imageSrc1 = load_data.load_image_c3(load_data.image_star, dsize=(128, 128))
    imageSrc2 = load_data.load_image_c3(load_data.image_bird, dsize=(128, 128))
    
    IC = ImageCompressionBasedOnData(n_color=16)

    # IC.fit(imageSrc1, mode='Euclidean')
    # imageTransformed = IC.transform(imageSrc2)
    # cv2.imwrite('./data/star_to_bird_Euclidean.png', imageTransformed)

    # IC.fit(imageSrc1, mode='Cosine')
    # imageTransformed = IC.transform(imageSrc2)
    # cv2.imwrite('./data/star_to_bird_Cosine.png', imageTransformed)

    IC.fit(imageSrc2, mode='Euclidean')
    imageTransformed = IC.transform(imageSrc1)
    cv2.imwrite('./data/bird_to_star_Euclidean.png', imageTransformed)

    # cv2.waitKey(0)
