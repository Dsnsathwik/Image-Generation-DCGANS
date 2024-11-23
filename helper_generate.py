import torch
import torchvision
import cv2
import numpy as np

def generate_imgs(model_path, DCGAN, num_imgs=500, batch_size=128, save=True):

    CUDA_DEVICE_NUM = 0
    DEVICE = torch.device(f'cuda:{CUDA_DEVICE_NUM}' if torch.cuda.is_available() else 'cpu')
    model = DCGAN().to(DEVICE)
    model.load_state_dict(torch.load('dcgan_celeba.pt'))
    idx = 1

    for i in range(2):
        noise = torch.randn(batch_size, 100, 1, 1, device=DEVICE)
    
        with torch.no_grad():
            gen_imgs = model.generator_forward(noise)

        for img in gen_imgs:
            img_np = img.cpu().numpy().transpose(1, 2, 0)
        
            # De-normalize if necessary (assuming images are in range [-1, 1])
            img_np = (img_np + 1) * 127.5  # Converts [-1, 1] to [0, 255]
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        
            img_np = img_np.astype(np.uint8)
            cv2.imwrite(f'./results/generated_imgs/gen_img_{idx}.jpg', img_np)
        
            idx += 1