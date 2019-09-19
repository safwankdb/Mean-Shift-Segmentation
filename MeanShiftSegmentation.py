import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image


def mean_shift(img, h=0.1, n=10):
    H, W = img.shape[:2]
    img1 = img.copy()
    img2 = img.copy()
    for i in range(n):
        for y in range(H):
            for x in range(W):
                k = np.square((img1[y, x]-img1)/h).sum(-1)
                k = np.exp(-k)
                a = ((img1-img1[y, x]) * np.expand_dims(k, -1)).sum((0, 1))
                grad = a/k.sum()
                img2[y, x] += grad
            print('Progress: {:.02f}{}'.format(
                100*(i*H+y+1)/n/H, '%'), end='\r')
        img1 = img2
    return img1


if __name__ == "__main__":
    img = Image.open('imgs/test.jpg').resize((200, 150))
    img = np.array(img)/255
    img_s = mean_shift(img, h=0.1, n=15)

    plt.figure(figsize=(20, 10))
    plt.subplot(121)
    plt.imshow(img)
    plt.subplot(122)
    plt.imshow(img_s)
    plt.tight_layout()
    plt.savefig('a.jpg')
    plt.show()

    colors = img.reshape(-1, 3)
    plt.figure(figsize=(20, 10))
    plt.subplot(121, projection='3d')
    plt.scatter(colors[:, 0], colors[:, 1], colors[:, 2], c=colors)

    colors = img_s.reshape(-1, 3)
    plt.subplot(122, projection='3d')
    plt.scatter(colors[:, 0], colors[:, 1], colors[:, 2], c=colors)
    plt.tight_layout()
    plt.show()
