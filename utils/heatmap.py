import numpy
import math


# heatmap 生成函数 和 heatmap refine 函数

'''
热图 heatmap 生成的几种实现函数, ThreeSigmaGaussian 最快

img_height : 生成 heatmap 的高度
img_width : 生成 heatmap 的宽度
c_x : 点的横坐标,对应 heatmap 的宽
c_y : 点的纵坐标,对应 heatmap 的高
sigma : heatmap kernel size

'''
def gene_heatmap(label, sigma):
    '''
    添加指定 heatmap 生成函数 
    '''
    # assert label.dim() == 5

    heatmap = []
    img_h, img_w = 120, 80
    for batch_size in range(label.shape[0]):
        for joint_size in range(label.shape[1]):
            [x, y] = label[batch_size][joint_size]
            heatmap.append(ThreeSigmaGaussian(img_h, img_w, numpy.array(x), numpy.array(y), sigma))
    heatmap = numpy.array(heatmap).reshape(label.shape[0], label.shape[1], img_h, img_w)
    return heatmap
dir

def ThreeSigmaGaussian(img_height, img_width, c_x, c_y, sigma):
    '''
    img_height : 生成 heatmap 的高度
    img_width : 生成 heatmap 的宽度
    c_x : 点的横坐标,对应 heatmap 的宽
    c_y : 点的纵坐标,对应 heatmap 的高
    sigma : heatmap kernel size

    '''
    heatmap = numpy.zeros((img_height, img_width))
    tmp_size = sigma * 3
    ul = [int(c_x - tmp_size), int(c_y - tmp_size)]
    br = [int(c_x + tmp_size + 1), int(c_y + tmp_size + 1)]
    size = 2 * tmp_size + 1
    x = numpy.arange(0, size, 1, numpy.float32)
    y = x[:, numpy.newaxis]
    x0 = y0 = size // 2
    exponent = ((x - x0)**2 + (y - y0)**2) / (2 * (sigma ** 2))
    g = numpy.exp(-exponent)
    # usable gaussian range
    g_x = max(0, -ul[0]), min(br[0], img_width) - ul[0]
    g_y = max(0, -ul[1]), min(br[1], img_height) - ul[1]
    # image range
    img_x = max(0, ul[0]), min(br[0], img_width)
    img_y = max(0, ul[1]), min(br[1], img_height)
    heatmap[img_y[0]:img_y[1], img_x[0]:img_x[1]] = g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
    # heatmap = target.astype(numpy.uint8)
    # heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_HOT)
    return heatmap

def CenterLabelHeatMap(img_height, img_width, c_x, c_y, sigma):
    X1 = numpy.linspace(1, img_width, img_width)
    Y1 = numpy.linspace(1, img_height, img_height)
    [X, Y] = numpy.meshgrid(X1, Y1)
    X = X - c_x
    Y = Y - c_y
    D2 = X * X + Y * Y
    E2 = 2.0 * sigma * sigma
    Exponent = D2 / E2
    heatmap = numpy.exp(-Exponent)
    return heatmap

def CenterGaussianHeatMap(img_height, img_width, c_x, c_y, variance):
    gaussian_map = numpy.zeros((img_height, img_width))
    for x_p in range(img_width):
        for y_p in range(img_height):
            dist_sq = (x_p - c_x)**2 + (y_p - c_y)**2
            exponent = dist_sq / 2.0 /variance / variance
            gaussian_map[y_p, x_p] = numpy.exp(-exponent)
    return gaussian_map

'''
heatmap refine 函数
'''
def hm_kernel_size(hm_type, current_epoch, threshold=4):
    '''
    refine heatmap kernel size during traing
    based on current epoch
    heatmap_type = ['static', 'stage', 'liner', 'exp', 'new_exp']
    '''
    assert hm_type in range(5)
    
    heatmap_type = ['static', 'stage', 'liner', 'exp', 'new_exp']

    if heatmap_type[hm_type] == heatmap_type[0]:
        kernel_size = 7

    elif heatmap_type[hm_type] == heatmap_type[1]:
        # stage. parameter: k, b
        k, a, b = -40, 2, 8
        kernel_size = int(current_epoch/k)*a + b

    elif heatmap_type[hm_type] == heatmap_type[2]:
        # linear. parameter: k, b
        k, b = -1/10, 8
        kernel_size = k*(current_epoch+1) + b

    elif heatmap_type[hm_type] == heatmap_type[3]:
        # exp. parameter: [alpha beta k] or [a b k]
        a, b, k = -1/110, 1, 3
        kernel_size = (math.exp(a*(current_epoch-1))+b)**k

    elif heatmap_type[hm_type] == heatmap_type[4]:
        kernel_size = (math.exp((29-(current_epoch))/20))+4

    return max(kernel_size, threshold)



if __name__ == '__main__':
    
    import time
    import cv2
    import numpy as np
    import matplotlib.pyplot as plt

    height, width = 400, 200
    c_y, c_x = 200, 100


    start = time.time()
    heatmap1 = ThreeSigmaGaussian(height, width, c_x, c_y, 10)
    t1 = time.time() - start

    start = time.time()
    heatmap2 = CenterLabelHeatMap(height, width, c_x, c_y, 10)
    t2 = time.time() - start

    start = time.time()
    heatmap3 = CenterGaussianHeatMap(height, width, c_x, c_y, 10)
    t3 = time.time() - start

    print(t1, t2, t3)

    plt.subplot(1, 3, 1)
    plt.imshow(heatmap1)
    plt.subplot(1, 3, 2)
    plt.imshow(heatmap2)
    plt.subplot(1, 3, 3)
    plt.imshow(heatmap3)
    plt.show()

    print('End.')

    k_size = hm_kernel_size(4, 50)
    print(k_size)