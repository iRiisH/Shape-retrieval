import os
import cv2
import numpy as np


def get_imgs(n_views):
    """
    returns paths of the images
    """
    root_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    img = os.path.join(root_dir, 'data/imgs/m{}/m{}_{}.png'.format(0, 0, 0))
    return img


def mask_image(img):
    m, n = img.shape[:2]
    res = np.zeros((m, n), dtype=img.dtype)
    for i in range(m):
        for j in range(n):
            if np.array_equal(img[i, j], [255, 255, 255]):
                res[i, j] = 255
            else:
                res[i, j] = 0
    # cv2.imshow('mask', res)
    # cv2.waitKey(0)
    return res


def compute_a(mask_img):
    """
    compute 'feature a', ie. silhouette length relative to image area
    """
    m, n = mask_img.shape[:2]
    length = 0
    for i in range(m):
        for j in range(n):
            dx = 0
            if i < m-1:
                dx = float(mask_img[i+1, j] - mask_img[i, j])
            dy = 0
            if j < n-1:
                dy = float(mask_img[i, j+1] - mask_img[i, j])
            grad = dx*dx+dy*dy
            if grad > 0:
                length += 1
    return float(length)/(m*n)


def compute_b(mask_img):
    """
    compute 'feature b', ie. projected area relative to image area
    """
    m, n = mask_img.shape[:2]
    area = 0
    for i in range(m):
        for j in range(n):
            area += 1
    return float(area)/(m*n)

if __name__ == '__main__':
    n_views = 7
    img_path = get_imgs(n_views)
    img = cv2.imread(img_path)
    mask = mask_image(img)
    compute_a(mask)
