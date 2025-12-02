# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""分析扩散过程生成的噪声特性，验证其是否接近纯高斯噪声"""

import tqdm
import pickle
import numpy as np
import torch
import PIL.Image
import dnnlib
import matplotlib.pyplot as plt
import os
from scipy import stats
import seaborn as sns

device = torch.device('cuda')

def load_image(image_path, device):
    """加载图像并转换为张量"""
    img = PIL.Image.open(image_path).convert('RGB')
    img = np.array(img)
    img = torch.from_numpy(img).permute(2, 0, 1).float().to(device)
    # 归一化到 [-1, 1] 范围
    img = (img - 127.5) / 127.5
    return img.unsqueeze(0)  # 添加批次维度

def ode_forward_diffusion(x0, net, num_steps=18, sigma_min=0.002, sigma_max=80, rho=7, device=torch.device('cuda')):
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
        denoised = net(x_current, t_cur).to(torch.float64)
        
        # 计算ODE的导数（去噪方向）
        d_cur = (x_current - denoised) / t_cur
        
        # 反向ODE步骤（加噪方向）
        x_next = x_current + (t_next - t_cur) * d_cur
        
        images.append(x_next.detach().cpu())
        x_current = x_next
    
    return images, t_steps

def ode_reverse_diffusion(noisy_x, net, num_steps=18, sigma_min=0.002, sigma_max=80, rho=7, device=torch.device('cuda')):
    """使用ODE进行逆向扩散过程：从噪声恢复图像"""
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=device)
    x_next = noisy_x.to(torch.float64).to(device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    print("reverse diffusion steps:", t_steps)
    # 存储中间结果
    images = [x_next.detach().cpu()]
    
    # 逆向扩散过程（ODE的正向过程）
    for i, (t_cur, t_next) in tqdm.tqdm(list(enumerate(zip(t_steps[:-1], t_steps[1:]))), unit='step', desc="逆向扩散"):
        x_cur = x_next
        
        # 欧拉步骤
        denoised = net(x_cur, t_cur).to(torch.float64)
        d_cur = (x_cur - denoised) / t_cur
        x_next = x_cur + (t_next - t_cur) * d_cur
        
        images.append(x_next.detach().cpu())
    
    return images

def save_image(images, path):
    """保存图像"""
    images = images.squeeze(0).permute(1, 2, 0).numpy()
    images = (images * 127.5 + 128).clip(0, 255).astype(np.uint8)
    PIL.Image.fromarray(images, 'RGB').save(path)

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

def analyze_noise_properties(noise_tensor, pure_gaussian_tensor, save_path):
    """分析噪声特性并与纯高斯噪声比较"""
    # 转换为numpy数组进行分析
    noise_np = noise_tensor.squeeze(0).permute(1, 2, 0).numpy()
    pure_gaussian_np = pure_gaussian_tensor.squeeze(0).permute(1, 2, 0).numpy()
    
    # 展平数组用于统计分析
    noise_flat = noise_np.flatten()
    pure_gaussian_flat = pure_gaussian_np.flatten()
    
    # 创建分析图表
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. 直方图比较
    axes[0, 0].hist(noise_flat, bins=50, alpha=0.7, label='扩散噪声', density=True)
    axes[0, 0].hist(pure_gaussian_flat, bins=50, alpha=0.7, label='纯高斯噪声', density=True)
    axes[0, 0].set_title('噪声分布比较')
    axes[0, 0].legend()
    axes[0, 0].set_xlabel('像素值')
    axes[0, 0].set_ylabel('密度')
    
    # 2. Q-Q图比较
    stats.probplot(noise_flat, dist="norm", plot=axes[0, 1])
    axes[0, 1].set_title('扩散噪声 Q-Q图')
    
    stats.probplot(pure_gaussian_flat, dist="norm", plot=axes[0, 2])
    axes[0, 2].set_title('纯高斯噪声 Q-Q图')
    
    # 3. 按通道分析
    colors = ['红色', '绿色', '蓝色']
    for i, color in enumerate(colors):
        channel_noise = noise_np[:, :, i].flatten()
        channel_gaussian = pure_gaussian_np[:, :, i].flatten()
        
        # 计算统计量
        noise_mean = np.mean(channel_noise)
        noise_std = np.std(channel_noise)
        noise_skew = stats.skew(channel_noise)
        noise_kurt = stats.kurtosis(channel_noise)
        
        gaussian_mean = np.mean(channel_gaussian)
        gaussian_std = np.std(channel_gaussian)
        gaussian_skew = stats.skew(channel_gaussian)
        gaussian_kurt = stats.kurtosis(channel_gaussian)
        
        # 打印统计信息
        print(f"{color}通道统计:")
        print(f"  扩散噪声: 均值={noise_mean:.4f}, 标准差={noise_std:.4f}, 偏度={noise_skew:.4f}, 峰度={noise_kurt:.4f}")
        print(f"  纯高斯噪声: 均值={gaussian_mean:.4f}, 标准差={gaussian_std:.4f}, 偏度={gaussian_skew:.4f}, 峰度={gaussian_kurt:.4f}")
        
        # 绘制分布比较
        axes[1, i].hist(channel_noise, bins=50, alpha=0.7, label='扩散噪声', density=True)
        axes[1, i].hist(channel_gaussian, bins=50, alpha=0.7, label='纯高斯噪声', density=True)
        axes[1, i].set_title(f'{color}通道噪声分布')
        axes[1, i].legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    
    # 计算整体统计差异
    noise_mean = np.mean(noise_flat)
    noise_std = np.std(noise_flat)
    noise_skew = stats.skew(noise_flat)
    noise_kurt = stats.kurtosis(noise_flat)
    
    gaussian_mean = np.mean(pure_gaussian_flat)
    gaussian_std = np.std(pure_gaussian_flat)
    gaussian_skew = stats.skew(pure_gaussian_flat)
    gaussian_kurt = stats.kurtosis(pure_gaussian_flat)
    
    print("\n整体统计比较:")
    print(f"扩散噪声: 均值={noise_mean:.4f}, 标准差={noise_std:.4f}, 偏度={noise_skew:.4f}, 峰度={noise_kurt:.4f}")
    print(f"纯高斯噪声: 均值={gaussian_mean:.4f}, 标准差={gaussian_std:.4f}, 偏度={gaussian_skew:.4f}, 峰度={gaussian_kurt:.4f}")
    
    # 计算Kolmogorov-Smirnov测试
    ks_statistic, ks_p_value = stats.ks_2samp(noise_flat, pure_gaussian_flat)
    print(f"\nKolmogorov-Smirnov测试:")
    print(f"统计量={ks_statistic:.4f}, p值={ks_p_value:.4f}")
    
    if ks_p_value > 0.05:
        print("结论: 不能拒绝两个分布相同的假设（p > 0.05）")
    else:
        print("结论: 两个分布显著不同（p <= 0.05）")
    
    return {
        'noise_mean': noise_mean,
        'noise_std': noise_std,
        'noise_skew': noise_skew,
        'noise_kurt': noise_kurt,
        'gaussian_mean': gaussian_mean,
        'gaussian_std': gaussian_std,
        'gaussian_skew': gaussian_skew,
        'gaussian_kurt': gaussian_kurt,
        'ks_statistic': ks_statistic,
        'ks_p_value': ks_p_value
    }

def analyze_spatial_correlation(noise_tensor, pure_gaussian_tensor, save_path):
    """分析噪声的空间相关性"""
    # 转换为numpy数组
    noise_np = noise_tensor.squeeze(0).permute(1, 2, 0).numpy()
    pure_gaussian_np = pure_gaussian_tensor.squeeze(0).permute(1, 2, 0).numpy()
    
    # 计算自相关函数
    def autocorr2d(arr):
        """计算2D自相关函数"""
        arr = arr - np.mean(arr)
        corr = np.fft.ifft2(np.fft.fft2(arr) * np.conj(np.fft.fft2(arr)))
        corr = np.fft.fftshift(corr)
        return corr / corr[arr.shape[0]//2, arr.shape[1]//2]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    colors = ['红色', '绿色', '蓝色']
    for i, color in enumerate(colors):
        # 计算自相关
        noise_autocorr = autocorr2d(noise_np[:, :, i])
        gaussian_autocorr = autocorr2d(pure_gaussian_np[:, :, i])
        
        # 显示中心区域的自相关
        center = noise_autocorr.shape[0] // 2
        window = 20
        noise_center = noise_autocorr[center-window:center+window, center-window:center+window]
        gaussian_center = gaussian_autocorr[center-window:center+window, center-window:center+window]
        
        # 绘制自相关图
        im1 = axes[0, i].imshow(noise_center, cmap='viridis', vmin=-0.1, vmax=1.0)
        axes[0, i].set_title(f'{color}通道扩散噪声自相关')
        plt.colorbar(im1, ax=axes[0, i])
        
        im2 = axes[1, i].imshow(gaussian_center, cmap='viridis', vmin=-0.1, vmax=1.0)
        axes[1, i].set_title(f'{color}通道纯高斯噪声自相关')
        plt.colorbar(im2, ax=axes[1, i])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def main():
    torch.manual_seed(1)
    model_root = 'https://nvlabs-fi-cdn.nvidia.com/edm/pretrained'
    
    network_pkl = f'{model_root}/edm-cifar10-32x32-cond-vp.pkl'
    
    index = 5
    # 使用项目中的示例图像
    image_path = f'testdata/test{index}.png'
    save_path = f'noise_analysis/test{index}'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    print(f'加载图像 "{image_path}"...')
    x0 = load_image(image_path, device)
    
    print(f'从 "{network_pkl}" 加载网络...')
    with dnnlib.util.open_url(network_pkl) as f:
        net = pickle.load(f)['ema'].to(device)
    
    # 前向扩散过程
    print("执行前向扩散过程...")
    x0_noise, _ = ode_forward_diffusion(x0, net, num_steps=18, sigma_min=0.002, sigma_max=20, rho=7, device=torch.device('cuda'))
    
    # 保存扩散噪声图像
    save_image(x0_noise[-1], os.path.join(save_path, 'diffusion_noise.png'))
    
    # 生成相同尺度的纯高斯噪声
    # 计算扩散噪声的标准差
    noise_std = torch.std(x0_noise[-1]).item()
    pure_gaussian = torch.randn_like(x0_noise[-1]) * noise_std
    
    # 保存纯高斯噪声图像
    save_image(pure_gaussian, os.path.join(save_path, 'pure_gaussian.png'))
    
    # 分析噪声特性
    print("分析噪声特性...")
    stats_results = analyze_noise_properties(
        x0_noise[-1], 
        pure_gaussian, 
        os.path.join(save_path, 'noise_properties_analysis.png')
    )
    
    # 分析空间相关性
    print("分析空间相关性...")
    analyze_spatial_correlation(
        x0_noise[-1], 
        pure_gaussian, 
        os.path.join(save_path, 'spatial_correlation_analysis.png')
    )
    
    # 保存统计结果
    with open(os.path.join(save_path, 'statistics_results.txt'), 'w') as f:
        f.write("噪声特性分析结果\n")
        f.write("==================\n\n")
        f.write(f"扩散噪声统计:\n")
        f.write(f"  均值={stats_results['noise_mean']:.4f}\n")
        f.write(f"  标准差={stats_results['noise_std']:.4f}\n")
        f.write(f"  偏度={stats_results['noise_skew']:.4f}\n")
        f.write(f"  峰度={stats_results['noise_kurt']:.4f}\n\n")
        f.write(f"纯高斯噪声统计:\n")
        f.write(f"  均值={stats_results['gaussian_mean']:.4f}\n")
        f.write(f"  标准差={stats_results['gaussian_std']:.4f}\n")
        f.write(f"  偏度={stats_results['gaussian_skew']:.4f}\n")
        f.write(f"  峰度={stats_results['gaussian_kurt']:.4f}\n\n")
        f.write(f"Kolmogorov-Smirnov测试:\n")
        f.write(f"  统计量={stats_results['ks_statistic']:.4f}\n")
        f.write(f"  p值={stats_results['ks_p_value']:.4f}\n")
        
        if stats_results['ks_p_value'] > 0.05:
            f.write("\n结论: 不能拒绝两个分布相同的假设（p > 0.05）\n")
            f.write("扩散噪声与纯高斯噪声在统计上无显著差异\n")
        else:
            f.write("\n结论: 两个分布显著不同（p <= 0.05）\n")
            f.write("扩散噪声与纯高斯噪声在统计上有显著差异\n")
    
    print(f"分析完成，结果已保存到 {save_path}")

if __name__ == "__main__":
    main()