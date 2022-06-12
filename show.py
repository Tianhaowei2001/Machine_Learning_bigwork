import matplotlib.pyplot as plt
import cv2
import pandas as pd
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()


def rle_encode(im):
    '''
    im: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels = im.flatten(order='F')
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

#对RLE码进行解码，转换成图片
def rle_decode(mask_rle, shape=(512, 512)):
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape, order='F')

# writer = SummaryWriter("log")
# writer.add_image("test_image",img)
# writer.close()


#联合二的结果0.9：1.1
subtt1 = pd.read_csv('result/subtt.csv', sep='\t', names=['name', 'mask'])
subtt1['name'] = subtt1['name'].apply(lambda x: './test_a/' + x)

test_mask_result = pd.read_csv('./result/unet.csv', sep='\t', names=['name', 'mask'])
test_mask_result['name'] = test_mask_result['name'].apply(lambda x: './test_a/' + x)
upp_mask = pd.read_csv('result/upp.csv', sep='\t', names=['name', 'mask'])
upp_mask['name'] = upp_mask['name'].apply(lambda x: './test_a/' + x)


def show_result():
    writer = SummaryWriter("resultlog")
    upp_mask = pd.read_csv('result/upp.csv', sep='\t', names=['name', 'mask'])
    for i in range(10):
        Upp = rle_decode(upp_mask['mask'].iloc[i])
        Upp[Upp > 0] = 254
        Upp = Upp.reshape(512,512,1)
        Upp = tf.convert_to_tensor(Upp)
        #one channel to three channels
        Upp = tf.image.grayscale_to_rgb(Upp)
        sess = tf.Session()
        Upp = sess.run(Upp)
        print(type(Upp))


        #numpy to tensor
        trans = transforms.ToTensor()
        Upp = trans(Upp)
        writer.add_image("test_image", Upp, i)

    writer.close()
    return None

# show_result()
def show_Four():
    data_img = cv2.imread(subtt1['name'].iloc[1])
    unet = rle_decode(test_mask_result['mask'].iloc[1])  # unet结果
    upp = rle_decode(upp_mask['mask'].iloc[1])  # upp结果
    subtt = rle_decode(subtt1['mask'].iloc[1])  # 联合2结果
    plt.subplot(141)
    plt.imshow(data_img)  # 原图
    plt.subplot(142)
    plt.imshow(subtt)  # 联合1
    plt.subplot(143)
    plt.imshow(unet)  # unet
    plt.subplot(144)
    plt.imshow(upp)  # upp
    plt.show()
    return None
def show_upp():
    upp = rle_decode(upp_mask['mask'].iloc[1])  # upp结果
    data_img = cv2.imread(subtt1['name'].iloc[1])   #原图
    plt.subplot(121)
    plt.imshow(data_img)
    plt.subplot(122)
    plt.imshow(upp)
    plt.show()

    return None

show_upp()

