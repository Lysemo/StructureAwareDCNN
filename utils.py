import torch
import torch.utils.data as data
import os
import cv2
import torch.nn.functional as F
import numpy as np
import argparse

class CriterionCalc():
    def __init__(self):
        pass
    def caclPSNR(self,image,gt):
        # image,gt belong to [0,255]
        if(torch.cuda.is_available()):
            image = image.cpu().numpy()
            gt = gt.cpu().numpy()
        else:
            image = image.numpy()
            gt = gt.numpy()
        MSE = np.mean(np.mean((image-gt)**2,-1),-1)
        PSNR = np.zeros(len(MSE))
        for i in range(len(MSE)):
            PSNR[i] = 10*np.log10(255**2/MSE[i])
        return np.mean(PSNR)
    def caclBatchPSNR(self,image,gt):
        # image,gt's shape [batchsize,c,h,w]
        PSNR = []
        for i in range(image.shape[0]):
            PSNR.append(self.caclPSNR(image[i],gt[i]))
        return np.mean(PSNR)

class Utils():
    def __init__(self):
        pass

    def saveGeneResult(self,img,path):
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        # img belong to [0,255]
        if(img.is_cuda):
            img = img.cpu().numpy()
        else:
            img = img.numpy()
        for im in img:
            im = np.transpose(im,(1,2,0))
            im = np.squeeze(im)
            cv2.imwrite(path,im)

class ImageDataset(data.Dataset):
    def __init__(self,data_dir,args):
        self.args = args
        self.image_paths = []
        self.gt_paths = []
        with open(data_dir+'/image_pair.txt','r') as f:
            lines = f.readlines()
        image_paths = [line.split(' ')[0] for line in lines]
        gt_paths = [line.split(' ')[1] for line in lines]
        self.image_paths = [os.path.join(data_dir,'image_gray',line.replace('\n','')) for line in image_paths]
        self.gt_paths = [os.path.join(data_dir,'gt_gray',line.replace('\n','')) for line in gt_paths]
        for p in self.gt_paths:
            if p.replace('gt','image') not in self.image_paths:
                raise Exception(p,'not agree with image paths')

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        gt_path = self.gt_paths[index]
        image = cv2.imread(image_path,0)
        gt = cv2.imread(gt_path,0)
        sobel_gt = cv2.Sobel(gt, cv2.CV_64F, 1, 1, ksize=3)
        h,w = image.shape
        r = np.random.randint(0, h-self.args['patch_size'])
        c = np.random.randint(0, w-self.args['patch_size'])

        patch_image = image[r:r+self.args['patch_size'],c:c+self.args['patch_size']]
        patch_gt = gt[r:r+self.args['patch_size'],c:c+self.args['patch_size']]
        patch_sobel_gt = sobel_gt[r:r+self.args['patch_size'],c:c+self.args['patch_size']]

        patch_image = np.reshape(patch_image,(self.args['patch_size'],self.args['patch_size'],1))
        patch_image = np.transpose(patch_image,(2,0,1))
        patch_gt = np.reshape(patch_gt,(self.args['patch_size'],self.args['patch_size'],1))
        patch_gt = np.transpose(patch_gt, (2, 0, 1))
        patch_sobel_gt = np.reshape(patch_sobel_gt, (self.args['patch_size'], self.args['patch_size'], 1))
        patch_sobel_gt = np.transpose(patch_sobel_gt, (2, 0, 1))

        patch_image = torch.FloatTensor(patch_image)
        patch_gt = torch.FloatTensor(patch_gt)
        patch_sobel_gt = torch.FloatTensor(patch_sobel_gt)

        # preprocess
        patch_image = patch_image / 127.5 - 1
        patch_gt = patch_gt / 127.5 - 1
        patch_sobel_gt = torch.tanh(patch_sobel_gt)

        return patch_image,patch_gt,patch_sobel_gt

    def __len__(self):
        return len(self.image_paths)

class ValImageDataset(data.Dataset):
    def __init__(self,data_dir,args):
        self.args = args
        self.image_paths = []
        self.gt_paths = []
        with open(data_dir+'/image_pair.txt','r') as f:
            lines = f.readlines()
        image_paths = [line.split(' ')[0] for line in lines]
        gt_paths = [line.split(' ')[1] for line in lines]
        self.image_paths = [os.path.join(data_dir,'image_gray',line.replace('\n','')) for line in image_paths]
        self.gt_paths = [os.path.join(data_dir,'gt_gray',line.replace('\n','')) for line in gt_paths]
        for p in self.gt_paths:
            if p.replace('gt','image') not in self.image_paths:
                raise Exception(p,'not agree with image paths')

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        gt_path = self.gt_paths[index]

        image = cv2.imread(image_path,0)
        gt = cv2.imread(gt_path,0)
        sobel_gt = cv2.Sobel(gt, cv2.CV_64F, 1, 1, ksize=3)

        image = np.reshape(image,(self.args['image_size'],self.args['image_size'],1))
        image = np.transpose(image,(2,0,1))
        gt = np.reshape(gt,(self.args['image_size'],self.args['image_size'],1))
        gt = np.transpose(gt, (2, 0, 1))
        sobel_gt = np.reshape(sobel_gt, (self.args['image_size'], self.args['image_size'], 1))
        sobel_gt = np.transpose(sobel_gt, (2, 0, 1))

        image = torch.FloatTensor(image)
        gt = torch.FloatTensor(gt)
        sobel_gt = torch.FloatTensor(sobel_gt)

        # preprocess
        image = image / 127.5 - 1
        gt = gt / 127.5 - 1
        sobel_gt = torch.tanh(sobel_gt)

        return image,gt,sobel_gt

    def __len__(self):
        return len(self.image_paths)

class ColorImageDataset(data.Dataset):
    def __init__(self,data_dir,args):
        self.args = args
        self.image_paths = []
        self.gt_paths = []
        with open(data_dir+'/image_pair.txt','r') as f:
            lines = f.readlines()
        image_paths = [line.split(' ')[0] for line in lines]
        gt_paths = [line.split(' ')[1] for line in lines]
        self.image_paths = [os.path.join(data_dir,'image',line.replace('\n','')) for line in image_paths]
        self.gt_paths = [os.path.join(data_dir,'gt',line.replace('\n','')) for line in gt_paths]
        for p in self.gt_paths:
            if p.replace('gt','image') not in self.image_paths:
                raise Exception(p,'not agree with image paths')

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        gt_path = self.gt_paths[index]

        image = cv2.imread(image_path)
        gt = cv2.imread(gt_path)
        sobel_gt = cv2.Sobel(gt, cv2.CV_64F, 1, 1, ksize=3)

        image = np.reshape(image, (self.args['image_size'], self.args['image_size'], 3))
        image = np.transpose(image, (2, 0, 1))
        gt = np.reshape(gt, (self.args['image_size'], self.args['image_size'], 3))
        gt = np.transpose(gt, (2, 0, 1))
        sobel_gt = np.reshape(sobel_gt, (self.args['image_size'], self.args['image_size'], 3))
        sobel_gt = np.transpose(sobel_gt, (2, 0, 1))

        image = torch.FloatTensor(image)
        gt = torch.FloatTensor(gt)
        sobel_gt = torch.FloatTensor(sobel_gt)

        # preprocess
        image = image / 127.5 - 1
        gt = gt / 127.5 - 1
        sobel_gt = torch.tanh(sobel_gt)

        return image, gt, sobel_gt, os.path.basename(image_path)

    def __len__(self):
        return len(self.image_paths)

def get_parser():
    parser = argparse.ArgumentParser('parameters')

    parser.add_argument('--batchsize',type=int,default=4,help='train batch')
    parser.add_argument('--epochs',type=int,default=200,help='epoch')
    parser.add_argument('--patch_size',type=int,default=32,help='image size')
    parser.add_argument('--lr',type=float,default=1e-5,help='lr')
    parser.add_argument('--num_works', type=int, default=8, help='num_works')
    parser.add_argument('--channels', type=int, default=1, help='channels')
    parser.add_argument('--res', type=str, default='./output', help='res')
    parser.add_argument('--model', type=str, default='./model', help='res')

    args = parser.parse_args()
    args = vars(args)
    return args

if __name__ == '__main__':
    args = get_parser()
    dataset = ImageDataset('./dataset/train', args)
    trainDataLoader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=args['batchsize'],
        shuffle=True,
        num_workers=args['num_works'],
        pin_memory=True,
    )
    for i,(image,gt,sobel_image,sobel_gt) in enumerate(trainDataLoader):
        print(image.shape,gt.shape,sobel_image.shape,sobel_gt.shape)
    pass








