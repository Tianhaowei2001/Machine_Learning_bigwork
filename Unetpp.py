import numpy as np
import pandas as pd
import pathlib, sys, os, random, time
import numba, cv2, gc
# from tqdm import tqdm_notebook
from tqdm import tqdm
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import KFold
import albumentations as A
import segmentation_models_pytorch as smp
import rasterio
from rasterio.windows import Window
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as D
import torchvision
from torchvision import transforms as T
from SegLoss.hausdorff import HausdorffDTLoss
from SegLoss.lovasz_loss import LovaszSoftmax



EPOCHES = 120
BATCH_SIZE = 2   #每轮训练，训练的图片数
IMAGE_SIZE = 512   #输入图片尺寸
# DEVICE = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')   #选择训练的位置
DEVICE = torch.device('cuda:0')   #利用GPU进行训练
import logging
#记录运行日志信息
logging.basicConfig(filename='./log/log_unetplusplus_sh_fold_3_continue2.log',
                    format='%(asctime)s - %(name)s - %(levelname)s -%(module)s:  %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S ',
                    level=logging.INFO)

#通过设置随机数种子，让表现较好的参数结果能够复现
def set_seeds(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seeds()

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
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return
    Returns numpy array, 1 - mask, 0 - background
    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape, order='F')

#进行训练集图片数据增强
train_trfm = A.Compose([
    # A.RandomCrop(NEW_SIZE*3, NEW_SIZE*3),
    A.Resize(IMAGE_SIZE, IMAGE_SIZE),    #修改图片尺寸
    A.HorizontalFlip(p=0.5),    #水平翻转，概率为0.5
    A.VerticalFlip(p=0.5),      #垂直翻转，概率为0.5
    A.RandomRotate90(),         #随机旋转90度
    A.OneOf([                     #oneof里的四种变换随机选择一种
        A.RandomContrast(),       #随机对比度
        A.RandomGamma(),          #随机伽马变化
        A.RandomBrightness(),      #随机亮度
        A.ColorJitter(brightness=0.07, contrast=0.07,
                      saturation=0.1, hue=0.1, always_apply=False, p=0.3),    #修改亮度对比度饱和度
    ], p=0.3),
])

#验证集数据增强
val_trfm = A.Compose([
    # A.CenterCrop(NEW_SIZE, NEW_SIZE),       #中心裁剪
    A.Resize(IMAGE_SIZE, IMAGE_SIZE),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(),
])

#load dataset
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

#读取训练集每个图片的RLE码
train_mask = pd.read_csv('./train_mask.csv', sep='\t', names=['name', 'mask'])
train_mask['name'] = train_mask['name'].apply(lambda x: './train/' + x)    #给图片名称添加前缀，变成图片路径

img = cv2.imread(train_mask['name'].iloc[0])     #根据第一张图片的路径将图片读取
mask = rle_decode(train_mask['mask'].iloc[0])    #对第一张图片的RLE码进行解码，将其解码成图片
plt.subplot(121)
plt.imshow(img)
plt.subplot(122)
plt.imshow(mask)
plt.show()
plt.close()
# print(rle_encode(mask) == train_mask['mask'].iloc[0])


#获取数据集
dataset = TianChiDataset(
    train_mask['name'][:7500].values,
    train_mask['mask'][:7500].fillna('').values,     #把目标值变成字符串格式
    train_trfm, False
)

skf = KFold(n_splits=5)    #交叉验证划分成五个子集
idx = np.array(range(len(dataset)))   #数据集里每个样本和标签进行编号



@torch.no_grad()
def validation(model, loader, loss_fn):
    losses = []
    model.eval()
    for image, target in loader:
        image, target = image.to(DEVICE), target.float().to(DEVICE)
        output = model(image)   #对图片进行训练
        loss = loss_fn(output, target)   #将结果与目标图进行评估损失
        losses.append(loss.item())    #将损失放入losses列表

    return np.array(losses).mean()   #返回损失的平均值


def np_dice_score(probability, mask):
    p = probability.reshape(-1)
    t = mask.reshape(-1)

    p = p > 0.5
    t = t > 0.5
    uion = p.sum() + t.sum()

    overlap = (p * t).sum()
    dice = 2 * overlap / (uion + 0.001)
    return dice


def validation_acc(model, val_loader, criterion):
    val_probability, val_mask = [], []
    model.eval()
    with torch.no_grad():
        for image, target in val_loader:
            image, target = image.to(DEVICE), target.float().to(DEVICE) #把图片装载到GPU上
            output = model(image)

            output_ny = output.sigmoid().data.cpu().numpy()
            target_np = target.data.cpu().numpy()

            val_probability.append(output_ny)
            val_mask.append(target_np)

    val_probability = np.concatenate(val_probability)
    val_mask = np.concatenate(val_mask)

    return np_dice_score(val_probability, val_mask)



class SoftDiceLoss(nn.Module):
    def __init__(self, smooth=1., dims=(-2, -1)):
        super(SoftDiceLoss, self).__init__()
        self.smooth = smooth
        self.dims = dims

    def forward(self, x, y):
        tp = (x * y).sum(self.dims)
        fp = (x * (1 - y)).sum(self.dims)
        fn = ((1 - x) * y).sum(self.dims)

        dc = (2 * tp + self.smooth) / (2 * tp + fp + fn + self.smooth)
        dc = dc.mean()
        return 1 - dc


bce_fn = nn.BCEWithLogitsLoss()
dice_fn = SoftDiceLoss()
#定义SoftMarginLoss
soft_margin_loss = nn.SoftMarginLoss()

def loss_fn(y_pred, y_true, ratio=0.8, hard=False):
    bce = bce_fn(y_pred, y_true)
    if hard:
        dice = dice_fn((y_pred.sigmoid()).float() > 0.5, y_true)
    else:
        dice = dice_fn(y_pred.sigmoid(), y_true)
    return ratio * bce + (1 - ratio) * dice


class Hausdorff_loss(nn.Module):
    def __init__(self):
        super(Hausdorff_loss, self).__init__()

    def forward(self, inputs, targets):
        return HausdorffDTLoss()(inputs, targets)


class Lovasz_loss(nn.Module):
    def __init__(self):
        super(Lovasz_loss, self).__init__()

    def forward(self, inputs, targets):
        return LovaszSoftmax()(inputs, targets)


criterion = HausdorffDTLoss()

header = r'''
        Train | Valid
Epoch |  Loss |  Loss | Time, m
'''
#          Epoch         metrics            time
raw_line = '{:6d}' + '\u2502{:7.4f}' * 2 + '\u2502{:6.2f}'
print(header)
logging.info(header)

for fold_idx, (train_idx, valid_idx) in enumerate(skf.split(idx, idx)):

    if fold_idx != 3:
        continue

    train_ds = D.Subset(dataset, train_idx)
    valid_ds = D.Subset(dataset, valid_idx)

    # define training and validation data loaders
    loader = D.DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    vloader = D.DataLoader(
        valid_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    fold_model_path = './round1/fold3_uppmodel_new3.pth'
    #定义uNet++网络模型
    model = smp.UnetPlusPlus(
        encoder_name="efficientnet-b4",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights=None,  # use `imagenet` pretreined weights for encoder initialization
        in_channels=3,  # model input channels (1 for grayscale images, 3 for RGB, etc.)
        classes=1,  # model output channels (number of classes in your dataset)
    )

    model.load_state_dict(torch.load(fold_model_path))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)   #选择优化器，lr表示学习率，weight_decay表示权重衰减系数
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=1, eta_min=1e-5, last_epoch=-1)

    model.to(DEVICE)

    best_loss = 10

    for epoch in range(1, EPOCHES + 1):
        losses = []
        start_time = time.time()
        #模型训练
        model.train()
        #对训练集图片的获取进行进度条追踪
        for image, target in tqdm(loader):
            image, target = image.to(DEVICE), target.float().to(DEVICE)
            optimizer.zero_grad()   #梯度初始值化为零
            output = model(image)   #对图像进行预测，向前传播获取预测值
            loss = loss_fn(output, target)    #求出loss
            # loss = criterion(output, target)
            loss.backward()    #反向传播梯度
            optimizer.step()   #更新所有参数
            losses.append(loss.item()) #记录损失
            # print(loss.item())

        vloss = validation(model, vloader, loss_fn)    #训练好的模型在验证集上进行验证
        scheduler.step(vloss)    #根据模型在验证集的表现对优化器的学习率进行更新
        logging.info(raw_line.format(epoch, np.array(losses).mean(), vloss,
                                     (time.time() - start_time) / 60 ** 1))   #进度条清零

        losses = []
        if vloss < best_loss:
            best_loss = vloss
            #保存模型参数
            torch.save(model.state_dict(), 'fold{}_uppmodel_new3.pth'.format(fold_idx))
            print("best loss is {}".format(best_loss))



