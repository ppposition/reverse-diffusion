import torch
import numpy as np
import pickle
import dnnlib
import os
from reverse_diffusion import ode_forward_diffusion, ode_reverse_diffusion, save_image, save_image_channel, save_image_grid

def main():
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载预训练模型
    model_root = 'https://nvlabs-fi-cdn.nvidia.com/edm/pretrained'
    network_pkl = f'{model_root}/edm-imagenet-64x64-cond-adm.pkl'
    
    with dnnlib.util.open_url(network_pkl) as f:
        net = pickle.load(f)['ema'].to(device)
    
    # 设置随机种子
    seed = 42
    torch.manual_seed(seed)
    
    # 创建输出目录
    save_dir = f"diffusion_demo/seed{seed}"
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. 从噪声生成图像
    print("从噪声生成图像...")
    noise = torch.randn(1, 3, 64, 64).to(device) * 20  # 初始噪声
    
    # 设置类别标签
    label = torch.zeros(net.label_dim, device=device).unsqueeze(0)
    label[0, 281] = 1  # 使用特定类别
    
    # 逆向扩散过程：从噪声到图像
    generated_images = ode_reverse_diffusion(
        noise, net, num_steps=18, sigma_min=0.002, sigma_max=20, rho=7, 
        device=device, label=label
    )
    generated_image = generated_images[-1]
    
    # 保存生成的图像
    save_image(generated_image, os.path.join(save_dir, 'generated.png'))
    print(f"生成的图像已保存到 {save_dir}/generated.png")
    
    # 2. 对生成图像加噪到指定时间步
    print("对生成图像加噪...")
    target_step = 12  # 指定时间步（中间步骤）
    
    # 前向扩散过程：从图像到噪声
    noisy_images, t_steps = ode_forward_diffusion(
        generated_image, net, num_steps=18, sigma_min=0.002, sigma_max=20, rho=7, 
        device=device, label=label
    )
    print("噪声图片数量:", len(noisy_images))
    # 获取指定时间步的噪声图像
    noisy_image = noisy_images[target_step]
    print(f"时间步 {target_step} 对应的噪声水平: {t_steps[target_step].item():.6f}")
    save_image(noisy_image, os.path.join(save_dir, f'noisy_step_{target_step}.png'))
    print(f"加噪图像已保存到 {save_dir}/noisy_step_{target_step}.png")
    
    # 3. 从指定时间步往回重建
    print("从指定时间步重建图像...")
    
    label = torch.zeros(net.label_dim, device=device).unsqueeze(0)
    label[0, 261] = 1  
    # 从噪声图像重建
    reconstructed_images = ode_reverse_diffusion(
        noisy_image, net, num_steps=18, sigma_min=0.002, sigma_max=20, rho=7, 
        device=device, label=label, special_steps=target_step
    )
    reconstructed_image = reconstructed_images[-1]
    save_image_grid(reconstructed_images, os.path.join(save_dir, 'reconstructed_grid.png'), grid_size=(4, 5))
    # 保存重建的图像
    save_image(reconstructed_image, os.path.join(save_dir, 'reconstructed.png'))
    print(f"重建的图像已保存到 {save_dir}/reconstructed.png")
    
    # 计算并打印原始生成图像与重建图像的差异
    diff = torch.abs(generated_image - reconstructed_image).mean().item()
    print(f"原始生成图像与重建图像的平均绝对差异: {diff:.6f}")

if __name__ == "__main__":
    main()