from reverse_diffusion import ode_forward_diffusion, ode_reverse_diffusion, save_image, save_image_channel, save_image_grid
import torch
import numpy as np
import PIL.Image
import tqdm
import dnnlib
import matplotlib.pyplot as plt
import pickle
import os

device = torch.device('cuda')

def main():
    model_root = 'https://nvlabs-fi-cdn.nvidia.com/edm/pretrained'
    
    network_pkl = f'{model_root}/edm-imagenet-64x64-cond-adm.pkl'
    
    #x0 = x0 + 0.2 * torch.randn_like(x0)
    with dnnlib.util.open_url(network_pkl) as f:
        net = pickle.load(f)['ema'].to(device)
    seed = 4
    torch.manual_seed(seed)
    save_dir = f"difference_result/seed{seed}" 
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    noisy_x = torch.randn(1, 3, 64, 64).to(device) * 20
    label = torch.zeros(net.label_dim, device=device).unsqueeze(0)
    label[0, 281] = 1  # 假设使用类别0
    np.save(os.path.join(save_dir, 'noisy_x.npy'), noisy_x.detach().cpu().numpy())
    save_image(noisy_x.detach().cpu(), os.path.join(save_dir, 'noisy_x.png'))
    x0 = ode_reverse_diffusion(noisy_x, net, num_steps=18, sigma_min=0.002, sigma_max=20, rho=7, device=torch.device('cuda'), label=label)[-1]
    np.save(os.path.join(save_dir, 'x0.npy'), x0.detach().cpu().numpy())
    save_image(x0, os.path.join(save_dir, 'x0.png'))
    x0_noise, _ = ode_forward_diffusion(x0, net, num_steps=18, sigma_min=0.002, sigma_max=20, rho=7, device=torch.device('cuda'), label=label)
    np.save(os.path.join(save_dir, 'x0_noise.npy'), x0_noise[-1].detach().cpu().numpy())
    save_image(x0_noise[-1], os.path.join(save_dir, 'x0_noise.png'))
    save_image_channel(x0_noise[-1], os.path.join(save_dir, 'x0_noise_channel.png'))
    save_image_grid(x0_noise, os.path.join(save_dir, 'x0_noise_grid.png'), grid_size=(4, 5))
    save_image(x0_noise[-1]-noisy_x.detach().cpu(), os.path.join(save_dir, 'difference.png'))
    save_image_channel(x0_noise[-1]-noisy_x.detach().cpu(), os.path.join(save_dir, 'difference_channel.png'))
    np.save(os.path.join(save_dir, 'difference.npy'), x0_noise[-1]-noisy_x.detach().cpu().numpy())
    label = torch.zeros(net.label_dim, device=device).unsqueeze(0)
    label[0, 280] = 1  
    reconstructed = ode_reverse_diffusion(x0_noise[-1], net, num_steps=18, sigma_min=0.002, sigma_max=20, rho=7, device=torch.device('cuda'), label=label)
    np.save(os.path.join(save_dir, 'reconstructed.npy'), reconstructed[-1].detach().cpu().numpy())
    save_image(reconstructed[-1], os.path.join(save_dir, 'reconstructed.png'))
    print(torch.max(reconstructed[-1].detach().cpu()-x0.detach().cpu()))
    save_image(reconstructed[-1].detach().cpu()-x0.detach().cpu(), os.path.join(save_dir, 'reconstructed_channel.png'))
    save_image_grid(reconstructed, os.path.join(save_dir, 'reconstructed_grid.png'), grid_size=(4, 5))

if __name__ == "__main__":
    main()