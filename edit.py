from reverse_diffusion import ode_forward_diffusion, ode_reverse_diffusion, save_image, save_image_channel, save_image_grid
import torch
import numpy as np
import PIL.Image
import tqdm
import dnnlib
import matplotlib.pyplot as plt
import pickle
import os
import argparse
device = torch.device('cuda')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--percentage', type=int, default=None, help='percentage of the image to edit')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for reproducibility')
    args = parser.parse_args()
    model_root = 'https://nvlabs-fi-cdn.nvidia.com/edm/pretrained'
    
    network_pkl = f'{model_root}/edm-afhqv2-64x64-uncond-vp.pkl'
    
    #x0 = x0 + 0.2 * torch.randn_like(x0)
    with dnnlib.util.open_url(network_pkl) as f:
        net = pickle.load(f)['ema'].to(device)
    torch.manual_seed(args.seed)
    save_dir = f"edit_result/seed{args.seed}" 
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    noisy_x = torch.randn(1, 3, 32, 32).to(device) * 20
    noisy_y = torch.randn(1, 3, 32, 32).to(device) * 20
    np.save(os.path.join(save_dir, 'noisy_x.npy'), noisy_x.detach().cpu().numpy())
    np.save(os.path.join(save_dir, 'noisy_y.npy'), noisy_y.detach().cpu().numpy())
    
    x0 = ode_reverse_diffusion(noisy_x, net, num_steps=18, sigma_min=0.002, sigma_max=20, rho=7, device=torch.device('cuda'))[-1]
    np.save(os.path.join(save_dir, 'x0.npy'), x0.detach().cpu().numpy())
    save_image(x0, os.path.join(save_dir, 'x0.png'))
    y0 = ode_reverse_diffusion(noisy_y, net, num_steps=18, sigma_min=0.002, sigma_max=20, rho=7, device=torch.device('cuda'))[-1]
    np.save(os.path.join(save_dir, 'y0.npy'), y0.detach().cpu().numpy())
    save_image(y0, os.path.join(save_dir, 'y0.png'))
    
    if args.percentage is None:
        images = []
        for percentage in np.linspace(0, 100, num=21):
            theta = torch.pi * torch.tensor(percentage / 200.0)
            mix_noise = torch.cos(theta) * noisy_x + torch.sin(theta) * noisy_y
            #save_image(mix_noise.detach().cpu(), os.path.join(save_dir, 'mix_noise.png'))
            np.save(os.path.join(save_dir, 'mix_noise.npy'), mix_noise.detach().cpu().numpy())
            mix_x0 = ode_reverse_diffusion(mix_noise, net, num_steps=18, sigma_min=0.002, sigma_max=20, rho=7, device=torch.device('cuda'))[-1]
            #np.save(os.path.join(save_dir, 'mix_x0.npy'), mix_x0.detach().cpu().numpy())
            #save_image(mix_x0, os.path.join(save_dir, 'mix_x0.png'))
            images.append(mix_x0)
        save_image_grid(torch.stack(images, dim=0), os.path.join(save_dir, 'edited_grid.png'))
    else:
        percentage = args.percentage
        theta = torch.pi * torch.tensor(percentage / 200.0)
        mix_noise = torch.cos(theta) * noisy_x + torch.sin(theta) * noisy_y
        save_image(mix_noise.detach().cpu(), os.path.join(save_dir, 'mix_noise.png'))
        np.save(os.path.join(save_dir, 'mix_noise.npy'), mix_noise.detach().cpu().numpy())
        mix_x0 = ode_reverse_diffusion(mix_noise, net, num_steps=18, sigma_min=0.002, sigma_max=20, rho=7, device=torch.device('cuda'))[-1]
        np.save(os.path.join(save_dir, 'mix_x0.npy'), mix_x0.detach().cpu().numpy())
        save_image(mix_x0, os.path.join(save_dir, 'mix_x0.png'))
if __name__ == "__main__":
    main()