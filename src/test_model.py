import torch
from glob import glob
from PIL import Image, ImageOps
import cv2 as cv
from lib.opts import opts
from lib.models.model import create_model, load_model
from lib.datasets.dataset_factory import get_dataset
import os
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
from lib.trains.train_factory import train_factory

def process_and_save_image(pred, save_path):
    if pred.max() <= 1.0:
        pred = (pred * 255).astype(np.uint8)  # 从 [0, 1] 转换到 [0, 255]

    # 黑白反转处理
    pred = np.where(pred == 0, 255, 0)  # 将 0 转换为 255，255 转换为 0
    # plt.imshow(pred, cmap='gray')
    # 添加边框
    pred = cv.copyMakeBorder(pred, 4, 225, 0, 0, 
                borderType=cv.BORDER_CONSTANT, 
                value=[0, 0, 0]  # 颜色为黑色 (BGR)
            )
    # 保存图像
    cv.imwrite(save_path, pred)  # 使用 OpenCV 保存图像
    plt.imsave(save_path, pred, cmap='gray')  # 使用 Matplotlib 保存图像

def val_img( opt) -> None:
    save_img_path = './dataset/output/'
      
    Dataset = get_dataset(opt.dataset, opt.task)
    opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
    opt.device = torch.device('cuda' if opt.gpus[0] >= 0 else 'cpu')
    
    model = create_model(opt.model_name)
    model = model.to(opt.device)
    model.eval() 

    val_loader = torch.utils.data.DataLoader(
        Dataset(opt, 'test'),
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    result=np.zeros([1024, 512], np.uint8)
    for i, inputs in enumerate(val_loader):
        with torch.no_grad():  
            output = model(inputs['input'].to(opt.device))
            pred = output.squeeze(0).squeeze(0)
            pred = pred.cpu()
            pred = pred.numpy()
            pred = cv.resize(pred, (512, 800))
            # plt.imshow(pred, cmap='gray')
            # pred = cv.copyMakeBorder(pred, 4, 225, 0, 0, 
            #     borderType=cv.BORDER_CONSTANT, 
            #     value=[0, 0, 0]  # 颜色为黑色 (BGR)
            # )
            original_file_name = val_loader.dataset.images[i]  # 假设 'images' 包含原始文件名
            relative_path = os.path.relpath(original_file_name, opt.data_root)
            save_path = os.path.join(save_img_path, relative_path)
            output_dir = os.path.dirname(save_path)
            os.makedirs(output_dir, exist_ok=True)


            process_and_save_image(pred, save_path)
            print(f'Saved result to {save_path}')
    # plt.show()
if __name__ == "__main__":
    opt = opts().parse()
    val_img( opt=opt)