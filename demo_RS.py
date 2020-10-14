import argparse
import torch
import torch.utils.data
from model import RS
import numpy as np
import os
from tqdm import tqdm
from utils import CriterionCalc,ValImageDataset,Utils,ColorImageDataset

def get_parser():
    parser = argparse.ArgumentParser('parameters')

    parser.add_argument('--patch_size',type=int,default=32,help='patch size')
    parser.add_argument('--image_size', type=int, default=256, help='image size')
    parser.add_argument('--num_works', type=int, default=8, help='num_works')
    parser.add_argument('--res', type=str, default='./output', help='res save path')
    parser.add_argument('--model', type=str, default='./model', help='model load path')

    args = parser.parse_args()
    args = vars(args)
    return args


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
args = get_parser()

def demo_gray():
    testDataset = ValImageDataset('./dataset/classical',args)
    testDataLoader = torch.utils.data.DataLoader(
        dataset=testDataset,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        pin_memory=True,
    )
    model = RS(16).to(device)
    state_dict = torch.load('./model/RS_best_psnr.pkl', map_location=device)
    model.load_state_dict(state_dict['model'], strict=False)
    print('RS model load done, epoch:%d, best_loss:%f, best psnr:%f'
          % (state_dict['epoch'], state_dict['best_loss'], state_dict['best_psnr']))

    model.eval()
    PSNR = []
    with torch.no_grad():
        for i, (image, gt, _) in tqdm(enumerate(testDataLoader),total=len(testDataLoader)):
            image,gt = image.to(device),gt.to(device)
            _,pre = model(image)
            pre = torch.tanh(pre)
            psnr = criCalc.caclBatchPSNR((pre+1)*127.5,(gt+1)*127.5)
            PSNR.append(psnr)
            utils.saveGeneResult((pre+1)*127.5,os.path.join(args['res'],'classical',str(i)+'_'+str(psnr)+'.png'))
        print('average psnr:%f' % (np.mean(PSNR)))

def demo_color():
    testDataset = ColorImageDataset('./dataset/test3', args)
    testDataLoader = torch.utils.data.DataLoader(
        dataset=testDataset,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        pin_memory=True,
    )
    model = RS(16).to(device)
    state_dict = torch.load('./model/RS_best_psnr.pkl', map_location=device)
    model.load_state_dict(state_dict['model'], strict=False)
    print('RS model load done, epoch:%d, best_loss:%f, best psnr:%f'
          % (state_dict['epoch'], state_dict['best_loss'], state_dict['best_psnr']))

    model.eval()
    PSNR = []
    with torch.no_grad():
        for i, (image, gt, _, name) in tqdm(enumerate(testDataLoader), total=len(testDataLoader)):
            image = image.to(device)
            psnr = []
            res = []
            for c in range(3):
                imagemeta = image[:,c,:,:]
                imagemeta = torch.reshape(imagemeta,(-1,1,args['image_size'],args['image_size']))
                gtmeta = gt[:,c,:,:]
                gtmeta = torch.reshape(gtmeta,(-1,1,args['image_size'],args['image_size']))
                _,pre = model(imagemeta)
                pre = torch.tanh(pre)
                res.append(pre)
                psnrmeta = criCalc.caclBatchPSNR((pre+1)*127.5,(gtmeta+1)*127.5)
                psnr.append(psnrmeta)
            output_color = torch.cat((res[0],res[1],res[2]),1)
            utils.saveGeneResult((output_color+1)*127.5, os.path.join(args['res'], 'test3', name[0]))
            PSNR.append(np.mean(psnr))
        print('average psnr:%f' % (np.mean(PSNR)))


if __name__ == '__main__':
    criCalc = CriterionCalc()
    utils = Utils()
    demo_color()
    # demo_gray()