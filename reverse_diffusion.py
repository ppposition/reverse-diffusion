# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""逆向扩散过程示例：使用ODE从原图像到噪声再回到原图像"""

import tqdm
import pickle
import numpy as np
import torch
import PIL.Image
import dnnlib
import matplotlib.pyplot as plt
import os
device = torch.device('cuda')

def load_image(image_path, device):
    """加载图像并转换为张量"""
    img = PIL.Image.open(image_path).convert('RGB')
    img = np.array(img)
    img = torch.from_numpy(img).permute(2, 0, 1).float().to(device)
    # 归一化到 [-1, 1] 范围
    img = (img - 127.5) / 127.5
    return img.unsqueeze(0)  # 添加批次维度

def ode_forward_diffusion(x0, net, num_steps=18, sigma_min=0.002, sigma_max=80, rho=7, device=torch.device('cuda'), label=None):
    """使用ODE进行前向扩散过程：从原图像到纯高斯噪声"""
    # 时间步离散化
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=device)
    t_steps = (sigma_min ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_max ** (1 / rho) - sigma_min ** (1 / rho))) ** rho
    #t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])])  # t_N = 0
    print("forward diffusion steps:", t_steps)
    # 存储中间结果
    images = [x0.detach().cpu()]
    x_current = x0.to(torch.float64).to(device)
    
    # 前向扩散过程（ODE的逆向过程）
    for i, (t_cur, t_next) in tqdm.tqdm(list(enumerate(zip(t_steps[:-1], t_steps[1:]))), unit='step', desc="前向扩散"):
        # 获取去噪后的结果
        denoised = net(x_current, t_cur, label).to(torch.float64)
        
        # 计算ODE的导数（去噪方向）
        d_cur = (x_current - denoised) / t_cur
        
        # 反向ODE步骤（加噪方向）
        x_next = x_current + (t_next - t_cur) * d_cur
        
        images.append(x_next.detach().cpu())
        x_current = x_next
    
    return images, t_steps

def ode_reverse_diffusion(noisy_x, net,num_steps=18, sigma_min=0.002, sigma_max=80, rho=7, device=torch.device('cuda'), label=None, special_steps=None):
    """使用ODE进行逆向扩散过程：从噪声恢复图像"""
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=device)
    x_next = noisy_x.to(torch.float64).to(device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])])
    # 存储中间结果
    images = [x_next.detach().cpu()]
    if special_steps is not None:
        t_steps = t_steps[(num_steps-special_steps-1):]
    print("reverse diffusion steps:", t_steps)
    # 逆向扩散过程（ODE的正向过程）
    for i, (t_cur, t_next) in tqdm.tqdm(list(enumerate(zip(t_steps[:-1], t_steps[1:]))), unit='step', desc="逆向扩散"):
        x_cur = x_next
        
        # 临时增加噪声
        #gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
        #t_hat = net.round_sigma(t_cur + gamma * t_cur)
        #x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * torch.randn_like(x_cur)
        
        # 欧拉步骤
        denoised = net(x_cur, t_cur, label).to(torch.float64)
        d_cur = (x_cur - denoised) / t_cur
        x_next = x_cur + (t_next - t_cur) * d_cur
        
        # 应用二阶校正
        '''if i < num_steps - 2:
            denoised = net(x_next, t_next, None).to(torch.float64)
            d_prime = (x_next - denoised) / t_next
            x_next = x_cur + (t_next - t_cur) * (0.5 * d_cur + 0.5 * d_prime)'''
        
        images.append(x_next.detach().cpu())
    
    return images

def save_image(images, path):
    """保存图像"""
    images = images.squeeze(0).permute(1, 2, 0).numpy()
    images = (images * 127.5 + 128).clip(0, 255).astype(np.uint8)
    PIL.Image.fromarray(images, 'RGB').save(path)

def save_image_channel(images, path):
    """保存图像通道"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for i in range(3):
        axes[i].imshow(images[0, i].numpy())
        axes[i].axis('off')
    plt.savefig(path)
      
def save_image_grid(images, path, grid_size=(4, 5)):
    """保存图像网格"""
    grid_h, grid_w = grid_size
    img_h, img_w = images[0].shape[2], images[0].shape[3]
    img_c = images[0].shape[1]
    
    # 创建网格
    grid = np.zeros((grid_h * img_h, grid_w * img_w, img_c), dtype=np.uint8)
    
    for i, img in enumerate(images[:grid_h * grid_w]):
        img = img.squeeze(0).permute(1, 2, 0).numpy()
        img = (img * 127.5 + 128).clip(0, 255).astype(np.uint8)
        
        row = i // grid_w
        col = i % grid_w
        grid[row * img_h:(row + 1) * img_h, col * img_w:(col + 1) * img_w] = img
    
    PIL.Image.fromarray(grid, 'RGB').save(path)

def reverse_process_demo(network_pkl, image_path, output_prefix,
                        num_steps=18, sigma_min=0.002, sigma_max=80, rho=7,
                        S_churn=0, S_min=0, S_max=float('inf'), S_noise=1,
                        device=torch.device('cuda')):
    """完整的逆向扩散过程演示"""
    
    # 检查CUDA可用性
    if device.type == 'cuda' and not torch.cuda.is_available():
        print("CUDA不可用，切换到CPU")
        device = torch.device('cpu')
    
    # 加载网络
    print(f'从 "{network_pkl}" 加载网络...')
    with dnnlib.util.open_url(network_pkl) as f:
        net = pickle.load(f)['ema'].to(device)
    
    # 调整噪声级别
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)
    
    # 加载图像
    print(f'加载图像 "{image_path}"...')
    x0 = load_image(image_path, device)
    
    # 如果需要，调整图像大小
    if x0.shape[2] != net.img_resolution or x0.shape[3] != net.img_resolution:
        from torchvision import transforms
        transform = transforms.Resize((net.img_resolution, net.img_resolution))
        x0 = transform(x0)
    
    # 获取类别标签（如果需要）
    class_labels = None
    if net.label_dim:
        # 假设是 CIFAR-10，使用第一个类别
        class_labels = torch.zeros(1, net.label_dim, device=device)
        class_labels[0, 0] = 1
    
    # 前向扩散过程（ODE逆向）
    print("执行前向扩散过程（ODE逆向）...")
    forward_images, t_steps = ode_forward_diffusion(x0, net, num_steps, sigma_min, sigma_max, rho, device)
    
    # 保存前向扩散结果
    save_image_grid(forward_images, f"{output_prefix}_forward.png", grid_size=(4, 5))
    print(f"前向扩散结果已保存到 {output_prefix}_forward.png")
    
    # 逆向扩散过程（ODE正向）
    print("执行逆向扩散过程（ODE正向）...")
    # 确保最后一个前向扩散图像在正确的设备上
    last_forward = forward_images[-1].to(device)
    reverse_images = ode_reverse_diffusion(last_forward, net, t_steps, class_labels, S_churn, S_min, S_max, S_noise)
    
    # 保存逆向扩散结果
    save_image_grid(reverse_images, f"{output_prefix}_reverse.png", grid_size=(4, 5))
    print(f"逆向扩散结果已保存到 {output_prefix}_reverse.png")
    
    # 保存最终恢复的图像
    final_img = reverse_images[-1].squeeze(0).permute(1, 2, 0).numpy()
    final_img = (final_img * 127.5 + 128).clip(0, 255).astype(np.uint8)
    PIL.Image.fromarray(final_img, 'RGB').save(f"{output_prefix}_restored.png")
    print(f"最终恢复的图像已保存到 {output_prefix}_restored.png")

def main():
    torch.manual_seed(1)
    model_root = 'https://nvlabs-fi-cdn.nvidia.com/edm/pretrained'
    
    network_pkl = f'{model_root}/edm-ffhq-64x64-uncond-vp.pkl'
    index = 2
    # 使用项目中的示例图像
    image_path = f'testdata/test{index}.png'
    save_path = f'result/test{index}'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    x0 = load_image(image_path, device)
    np.save(os.path.join(save_path, 'x0.npy'), x0.detach().cpu().numpy())
    #x0 = x0 + 0.2 * torch.randn_like(x0)
    with dnnlib.util.open_url(network_pkl) as f:
        net = pickle.load(f)['ema'].to(device)
    # 前向扩散过程（ODE逆向）
    x0_noise, _ = ode_forward_diffusion(x0, net, num_steps=50, sigma_min=0.002, sigma_max=20, rho=7, device=torch.device('cuda'))
    np.save(os.path.join(save_path, 'x0_noise.npy'), x0_noise[-1].detach().cpu().numpy())
    save_image(x0_noise[-1], os.path.join(save_path, 'x0_noise.png'))
    save_image_channel(x0_noise[-1], os.path.join(save_path,'x0_noise_channel.png'))
    save_image_grid(x0_noise, os.path.join(save_path,'x0_noise_grid.png'), grid_size=(4, 5))
    reconstructed = ode_reverse_diffusion(x0_noise[-1], net, num_steps=50, sigma_min=0.002, sigma_max=20, rho=7, device=torch.device('cuda'))
    np.save(os.path.join(save_path, 'reconstructed.npy'), reconstructed[-1].detach().cpu().numpy())
    save_image(reconstructed[-1], os.path.join(save_path,'reconstructed.png'))
    save_image(reconstructed[-1].detach().cpu()-x0.detach().cpu(), os.path.join(save_path,'reconstructed_channel.png'))
    save_image_grid(reconstructed, os.path.join(save_path,'reconstructed_grid.png'), grid_size=(4, 5))
    r_noise, _ = ode_forward_diffusion(reconstructed[-1], net, num_steps=50, sigma_min=0.002, sigma_max=20, rho=7, device=torch.device('cuda'))
    save_image(r_noise[-1], os.path.join(save_path,'r_noise.png'))
    save_image_channel(r_noise[-1], os.path.join(save_path,'r_noise_channel.png'))
    save_image_grid(r_noise, os.path.join(save_path,'r_noise_grid.png'), grid_size=(4, 5))
    save_image(r_noise[-1]-x0_noise[-1], os.path.join(save_path,'noise_difference.png'))
    save_image_channel(r_noise[-1]-x0_noise[-1], os.path.join(save_path,'noise_difference_channel.png'))

if __name__ == "__main__":
    main()