import argparse
import torch
import torch.utils.data
from model import DCNN
import numpy as np
import logging
import os
from tqdm import tqdm
from loss import MseLoss
from utils import CriterionCalc,ImageDataset,ValImageDataset

def get_parser():
    parser = argparse.ArgumentParser('parameters')

    parser.add_argument('--batchsize',type=int,default=32,help='train batch')
    parser.add_argument('--epochs',type=int,default=200,help='epoch')
    parser.add_argument('--patch_size',type=int,default=32,help='patch size')
    parser.add_argument('--image_size', type=int, default=256, help='image size')
    parser.add_argument('--lr',type=float,default=1e-2,help='lr')
    parser.add_argument('--num_works', type=int, default=8, help='num_works')
    parser.add_argument('--channels', type=int, default=1, help='channels')
    parser.add_argument('--res', type=str, default='./output', help='res save path')
    parser.add_argument('--model', type=str, default='./model', help='model save path')

    args = parser.parse_args()
    args = vars(args)
    return args

def getLogger(logPath):
    if not os.path.exists(os.path.dirname(logPath)):
        os.makedirs(os.path.dirname(logPath))
    logger = logging.getLogger('info')
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('[%(levelname)s] [%(asctime)s] %(message)s')

    std_handler = logging.StreamHandler()
    std_handler.setLevel(logger.level)
    std_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(logPath)
    file_handler.setLevel(logger.level)
    file_handler.setFormatter(formatter)

    logger.addHandler(std_handler)
    logger.addHandler(file_handler)
    return logger

def adjust_learning_rate(optimizer, epoch):
    rate = np.power(0.8,epoch//5)
    lr = rate * args['lr']
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    # rate = np.power(1.0-epoch/float(args['epochs'] + 1),0.9)
    # lr = rate * args['lr']
    # for param_group in optimizer.param_groups:
    #     param_group['lr'] = lr
    return lr

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
args = get_parser()

def train():
    dataset = ImageDataset('./dataset/train', args)
    trainDataLoader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=args['batchsize'],
        shuffle=True,
        num_workers=args['num_works'],
        pin_memory=True,
    )
    val = ValImageDataset('./dataset/val', args)
    valDataLoader = torch.utils.data.DataLoader(
        dataset=val,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        pin_memory=True,
    )
    model = DCNN(16).to(device)
    optimizer = torch.optim.SGD(model.parameters(),lr=args['lr'],weight_decay=1e-4,momentum=0.9)
    criterion = MseLoss().to(device)
    best_loss = np.inf
    best_psnr = -1
    for epoch in range(args['epochs']):
        lr = adjust_learning_rate(optimizer, epoch)
        logger.info('Epoch<%d/%d> current lr:%f', epoch + 1, args['epochs'], lr)
        model.train()
        train_loss = []
        for i, (image, gt, _) in enumerate(trainDataLoader):
            image, gt = image.to(device), gt.to(device)
            pre = model(image)
            pre = torch.tanh(pre)
            loss = criterion(pre,gt)
            train_loss.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            logger.info('Epoch<%d/%d>|Step<%d/%d> avg loss:%.6f' %
                        (epoch + 1, args['epochs'], i + 1,
                         np.ceil(trainDataLoader.dataset.__len__() / args['batchsize']),
                         np.mean(train_loss)))
        model.eval()
        val_loss = []
        PSNR = []
        with torch.no_grad():
            for j,(image,gt,_) in tqdm(enumerate(valDataLoader),total=len(valDataLoader)):
                image,gt = image.to(device),gt.to(device)
                pre = model(image)
                pre = torch.tanh(pre)
                loss = criterion(pre,gt)
                val_loss.append(loss.item())
                PSNR.append(criCalc.caclBatchPSNR((pre+1)*127.5,(gt+1)*127.5))
            if (np.mean(val_loss) <= best_loss):
                best_loss = np.mean(val_loss)
                model_related_loss = {'model': model.state_dict(), 'epoch': epoch + 1, 'best_loss': best_loss,
                                      'best_psnr': -1}
                torch.save(model_related_loss, args['model'] + '/IRS_best_loss.pkl')
            if (np.mean(PSNR) >= best_psnr):
                best_psnr = np.mean(PSNR)
                model_related_psnr = {'model': model.state_dict(), 'epoch': epoch + 1, 'best_loss': -1,
                                      'best_psnr': best_psnr}
                torch.save(model_related_psnr, args['model'] + '/IRS_best_psnr.pkl')
            logger.info('Epoch<%d/%d> current loss:%.6f, best loss:%.6f, current PSNR:%f(max:%f|min:%f), best PSNR:%f' %
                        (epoch + 1, args['epochs'], np.mean(val_loss), best_loss, np.mean(PSNR), np.max(PSNR),
                         np.min(PSNR), best_psnr))

if __name__ == '__main__':
    logger = getLogger('./log/output.txt')
    criCalc = CriterionCalc()
    train()

