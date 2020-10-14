# StructureAwareDCNN
Pytorch Implementation of Inverse halftoning through structure-aware deep convolutional neural networks https://arxiv.org/abs/1905.00637

![](.\log\img\net.jpg)

## Dataset Format

- image_gray: place halftone gray image
- gt_gray: place related continous gray image
- image_pair.txt: pre row include halftone image name and continous image name, the delimiter is a space.

## USE

- pretrain ISR net: python main_IRS.py
- train: python main_RS.py
- demo(generate continous image): python demo_RS.py