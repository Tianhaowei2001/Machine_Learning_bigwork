from torch.utils.data import Dataset
import os,cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms as T
import pandas as pd
import torch.utils.data as D
import segmentation_models_pytorch as smp
#对图片进行RLE编码
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

class MyData(Dataset):
    def __init__(self,img_dict):
        self.img_dict = img_dict
        self.img_name = os.listdir(self.img_dict)

    def __getitem__(self, index):
        self.path = os.path.join(self.img_dict,self.img_name[index])
        img = cv2.imread(self.path)
        return img

    def __len__(self):
        return len(self.img_name)


class TianChiDataset(D.Dataset):
    def __init__(self, paths, rles, transform, test_mode=False):
        self.paths = paths      #图片路径
        self.rles = rles        #每张图片的RLE编码
        self.transform = transform     #每张图片的转换方式
        self.test_mode = test_mode     #是否为训练模式

        self.len = len(paths)   #返回路径数组的长度
        self.as_tensor = T.Compose([         #转化成tensor类
            T.ToPILImage(),                  #将张量转变成PIL图片，convert tensor to PIL image
            T.Resize(IMAGE_SIZE),            #缩放尺寸
            T.ToTensor(),                    #将图片转换成tensor类型
            T.Normalize([0.625, 0.448, 0.688],
                        [0.131, 0.177, 0.101]),      #利用均值和标准差归一化张量图片
        ])

    # get data operation
    def __getitem__(self, index):
        img = cv2.imread(self.paths[index])     #根据索引路径读取图片
        if not self.test_mode:
            mask = rle_decode(self.rles[index])      #把图片对应的RLE进行解码
            augments = self.transform(image=img, mask=mask)  #把训练的原图和RLE解码所得的图进行图片增强转换
            return self.as_tensor(augments['image']), augments['mask'][None]   #最后把图片增强过后的图片进行修剪转换成tensor类型返回
        else:
            return self.as_tensor(img), ''            #如果是测试模式，图片就没有RLE编码，因此只需要把图片返回即可

    def __len__(self):      #返回数据集的样本数量
        return self.len


IMAGE_SIZE = 512
test_mask = pd.read_csv('./material/test_a_samplesubmit.csv', sep='\t', names=['name', 'mask'])
test_mask['name'] = test_mask['name'].apply(lambda x: './test_a/' + x)
# train_mask['name'] = train_mask['name'].apply(lambda x: './train/' + x)

trfm = T.Compose([
    T.ToPILImage(),
    T.Resize(IMAGE_SIZE),
    T.ToTensor(),
    T.Normalize([0.625, 0.448, 0.688],
                [0.131, 0.177, 0.101]),
])
test_data = TianChiDataset(
    test_mask['name'].values,
    test_mask['mask'].fillna('').values,
    trfm,True
)
# print(test_data[0])
# img = test_data[0][0]
# print(img)
img1 = cv2.imread(test_mask['name'].iloc[0])
plt.imshow(img1)
plt.show()
# writer = SummaryWriter("log")
# writer.add_image("test_image",img)
# writer.close()

model = smp.Unet(
        encoder_name="efficientnet-b4",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights=None,  # use `imagenet` pretreined weights for encoder initialization
        in_channels=3,  # model input channels (1 for grayscale images, 3 for RGB, etc.)
        classes=1,  # model output channels (number of classes in your dataset)
)
# model.load_state_dict(torch.load("fold3_uppmodel_new3.pth"))


DEVICE = torch.device('cuda:0')   #利用GPU进行训练
from tqdm import tqdm
trfm = T.Compose([
    T.ToPILImage(),
    T.Resize(IMAGE_SIZE),
    T.ToTensor(),
    T.Normalize([0.625, 0.448, 0.688],
                [0.131, 0.177, 0.101]),
])

subm = []

model.load_state_dict(torch.load("unet/fold4_unet_model_new4_s.pth"),False)
model.eval()
model.to(DEVICE) #把模型加载到GPU上
test_mask = pd.read_csv('./material/test_a_samplesubmit.csv', sep='\t', names=['name', 'mask'])
test_mask['name'] = test_mask['name'].apply(lambda x: './test_a/' + x)

for idx, name in enumerate(tqdm(test_mask['name'].iloc[:])):
    image = cv2.imread(name)
    image = trfm(image)
    with torch.no_grad():
        image = image.to(DEVICE)[None]     #把图片记载到GPU
        score = model(image)[0][0]    #对图像里的每个像素点进行预测评分
        score_sigmoid = score.sigmoid().cpu().numpy()     #设置一个sigmoid函数
        score_sigmoid = (score_sigmoid > 0.5).astype(np.uint8)   #如何sigmod分数大于0.5就认为该块像素是属于建筑的
        score_sigmoid = cv2.resize(score_sigmoid, (512, 512), interpolation = cv2.INTER_CUBIC)   #对每块像素进行上色


        # break
    subm.append([name.split('/')[-1], rle_encode(score_sigmoid)])   #把预测结果进行RLE编码放入subm列表当中


# In[35]:


subm = pd.DataFrame(subm)
subm.to_csv('result/unet.csv', index=None, header=None, sep='\t')

# train_mask = pd.read_csv('./train_mask.csv', sep='\t', names=['name', 'mask'])
# train_mask['name'] = train_mask['name'].apply(lambda x: './train/' + x)    #给图片名称添加前缀，变成图片路径
# print(train_mask)

test_mask_result = pd.read_csv('./result/unet.csv', sep='\t', names=['name', 'mask'])
test_mask_result['name'] = test_mask_result['name'].apply(lambda x: './test_a/' + x)
test_img = cv2.imread(test_mask_result['name'].iloc[1])
predict = rle_decode(test_mask_result['mask'].iloc[1])
plt.subplot(121)
plt.imshow(test_img)
plt.subplot(122)
plt.imshow(predict)
plt.show()