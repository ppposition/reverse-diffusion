"""
EDM 模型在 LSUN Bedroom 数据集上的实验
先测试模型加载和随机采样

生成过程和反转过程函数说明：
=====================================

1. 生成过程（采样过程）:
   - 主要函数: sample_heun() 或 sample_euler() (在 karras_diffusion.py)
   - 流程: 初始噪声 x_T (sigma_max=80.0) → 逐步去噪 → 最终图像 x_0 (sigma_min=0.002)
   - 去噪函数: diffusion.denoise(model, x_t, sigma_t)
   - 使用: _sample_with_noise() 方法调用

2. 反转过程（编码过程）:
   - 主要函数: encode_image() (在 cm_edm.py)
   - 流程: 清晰图像 x_0 (sigma_start) → 逐步加噪 → 噪声图像 x_T (sigma_end)
   - 实现: 使用 Karras 噪声调度逐步添加噪声，不需要模型

详细说明请查看: process_explanation.md
"""
import sys
from PIL import Image
import os
import numpy as np
import torch
import lmdb
import io
from datetime import datetime
import matplotlib.pyplot as plt
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# 获取脚本所在目录的父目录（controlnet目录）
script_dir = os.path.dirname(os.path.abspath(__file__))
controlnet_dir = os.path.dirname(script_dir)
sys.path.insert(0, controlnet_dir)

import cm_edm
from cm.random_util import get_generator
from cm_edm import interpolate_linear, slerp
import analyse

def read_images_from_lmdb(lmdb_path, num_images=2):
    """
    从 LMDB 数据库中读取图片
    
    Args:
        lmdb_path: LMDB 数据库路径
        num_images: 要读取的图片数量
    
    Returns:
        images: PIL Image 对象列表
        keys: 图片的 key 列表
    """
    env = lmdb.open(lmdb_path, map_size=1099511627776, max_readers=100, readonly=True)
    images = []
    keys = []
    with env.begin(write=False) as txn:
        cursor = txn.cursor()
        count = 0
        for key, val in cursor:
            if count >= num_images:
                break
            img = Image.open(io.BytesIO(val))
            images.append(img)
            keys.append(key.decode('ascii') if isinstance(key, bytes) else str(key))
            count += 1
    return images, keys

def preprocess_image(pil_image, image_size=256, device='cuda'):
    """
    将 PIL Image 转换为模型需要的张量格式
    
    Args:
        pil_image: PIL Image 对象
        image_size: 目标图像尺寸
        device: 设备
    
    Returns:
        tensor: 形状为 (1, 3, H, W)，值域 [-1, 1] 的张量
    """
    # 调整大小并转换为 RGB
    img = pil_image.resize((image_size, image_size), Image.LANCZOS)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # 转换为 numpy array (H, W, 3)，值域 [0, 255]
    img_array = np.array(img).astype(np.float32)
    
    # 转换为张量并归一化到 [-1, 1]
    img_tensor = torch.from_numpy(img_array).permute(2, 0, 1) / 127.5 - 1.0  # (3, H, W)
    img_tensor = img_tensor.unsqueeze(0).to(device)  # (1, 3, H, W)
    
    return img_tensor

def tensor_to_image(tensor):
    """
    将张量转换为 PIL Image
    
    Args:
        tensor: 形状为 (1, 3, H, W) 或 (3, H, W)，值域 [-1, 1] 的张量
    
    Returns:
        PIL Image 对象
    """
    if tensor.dim() == 4:
        tensor = tensor[0]  # (3, H, W)
    
    # 转换为 numpy array
    img_array = (tensor.permute(1, 2, 0).cpu().numpy() + 1) * 127.5
    img_array = np.clip(img_array, 0, 255).astype(np.uint8)
    
    return Image.fromarray(img_array)

def visualize_comparison(original_img, noise_img, reconstructed_img, save_path, title="Comparison", image_size=256):
    """
    绘制真实图像-反转噪声-生成图像三张子图的对比图
    
    Args:
        original_img: 原始图像 (PIL Image 或 tensor)
        noise_img: 反转噪声图像 (PIL Image 或 tensor)
        reconstructed_img: 重建图像 (PIL Image 或 tensor)
        save_path: 保存路径
        title: 图片标题
        image_size: 图像尺寸（默认 256）
    """
    # 计算噪声的norm（如果noise_img是tensor）
    noise_norm = None
    if isinstance(noise_img, torch.Tensor):
        noise_norm = torch.norm(noise_img).item()
    elif isinstance(noise_img, Image.Image):
        # 如果已经是PIL Image，需要先转换为tensor来计算norm
        noise_tensor = preprocess_image(noise_img, image_size=image_size, device='cpu')
        noise_norm = torch.norm(noise_tensor).item()
    
    # 转换为 PIL Image（如果输入是 tensor）
    if isinstance(original_img, torch.Tensor):
        original_img = tensor_to_image(original_img)
    if isinstance(noise_img, torch.Tensor):
        noise_img = tensor_to_image(noise_img)
    if isinstance(reconstructed_img, torch.Tensor):
        reconstructed_img = tensor_to_image(reconstructed_img)
    
    # 确保所有图像尺寸一致（统一调整为 image_size）
    target_size = (image_size, image_size)
    if original_img.size != target_size:
        original_img = original_img.resize(target_size, Image.LANCZOS)
    if noise_img.size != target_size:
        noise_img = noise_img.resize(target_size, Image.LANCZOS)
    if reconstructed_img.size != target_size:
        reconstructed_img = reconstructed_img.resize(target_size, Image.LANCZOS)
    
    # 创建图形
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 绘制原始图像
    axes[0].imshow(original_img)
    axes[0].set_title('Original Image', fontsize=14)
    axes[0].axis('off')
    
    # 绘制反转噪声
    axes[1].imshow(noise_img)
    noise_title = 'Encoded Noise'
    if noise_norm is not None:
        noise_title += f' (norm: {noise_norm:.2f})'
    axes[1].set_title(noise_title, fontsize=14)
    axes[1].axis('off')
    
    # 绘制重建图像
    axes[2].imshow(reconstructed_img)
    axes[2].set_title('Reconstructed Image', fontsize=14)
    axes[2].axis('off')
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"对比图已保存到: {save_path}")

def visualize_comparison_with_noisy(original_img, noisy_img, noise_img, reconstructed_img, save_path, title="Comparison", perturbation_sigma=None, image_size=256):
    """
    绘制原始图像-加噪图像-反转噪声-重建图像四张子图的对比图
    
    Args:
        original_img: 原始图像 (PIL Image 或 tensor)
        noisy_img: 加噪后的图像 (PIL Image 或 tensor)
        noise_img: 反转噪声图像 (PIL Image 或 tensor)
        reconstructed_img: 重建图像 (PIL Image 或 tensor)
        save_path: 保存路径
        title: 图片标题
        perturbation_sigma: 微扰噪声水平（用于显示在标题中）
        image_size: 图像尺寸（默认 256）
    """
    # 计算噪声的norm（如果noise_img是tensor）
    noise_norm = None
    if isinstance(noise_img, torch.Tensor):
        noise_norm = torch.norm(noise_img).item()
    elif isinstance(noise_img, Image.Image):
        # 如果已经是PIL Image，需要先转换为tensor来计算norm
        noise_tensor = preprocess_image(noise_img, image_size=image_size, device='cpu')
        noise_norm = torch.norm(noise_tensor).item()
    
    # 转换为 PIL Image（如果输入是 tensor）
    if isinstance(original_img, torch.Tensor):
        original_img = tensor_to_image(original_img)
    if isinstance(noisy_img, torch.Tensor):
        noisy_img = tensor_to_image(noisy_img)
    if isinstance(noise_img, torch.Tensor):
        noise_img = tensor_to_image(noise_img)
    if isinstance(reconstructed_img, torch.Tensor):
        reconstructed_img = tensor_to_image(reconstructed_img)
    
    # 确保所有图像尺寸一致（统一调整为 image_size）
    target_size = (image_size, image_size)
    if original_img.size != target_size:
        original_img = original_img.resize(target_size, Image.LANCZOS)
    if noisy_img.size != target_size:
        noisy_img = noisy_img.resize(target_size, Image.LANCZOS)
    if noise_img.size != target_size:
        noise_img = noise_img.resize(target_size, Image.LANCZOS)
    if reconstructed_img.size != target_size:
        reconstructed_img = reconstructed_img.resize(target_size, Image.LANCZOS)
    
    # 创建图形
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    # 绘制原始图像
    axes[0].imshow(original_img)
    axes[0].set_title('Original Image', fontsize=14)
    axes[0].axis('off')
    
    # 绘制加噪图像
    axes[1].imshow(noisy_img)
    if perturbation_sigma is not None:
        axes[1].set_title(f'Noisy Image (sigma={perturbation_sigma})', fontsize=14)
    else:
        axes[1].set_title('Noisy Image', fontsize=14)
    axes[1].axis('off')
    
    # 绘制反转噪声
    axes[2].imshow(noise_img)
    noise_title = 'Encoded Noise'
    if noise_norm is not None:
        noise_title += f' (norm: {noise_norm:.2f})'
    axes[2].set_title(noise_title, fontsize=14)
    axes[2].axis('off')
    
    # 绘制重建图像
    axes[3].imshow(reconstructed_img)
    axes[3].set_title('Reconstructed Image', fontsize=14)
    axes[3].axis('off')
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"对比图已保存到: {save_path}")

def visualize_interpolation(noise1, noise2, original_img1, original_img2, em_manager, 
                            frac_list, interpolation_type, steps, sampler, sigma_min, sigma_max, rho,
                            save_path, title="Interpolation", image_size=256):
    """
    可视化插值结果：第一行显示插值图片，第二行显示对应的插值噪声
    
    Args:
        noise1: 第一个噪声张量 (1, 3, H, W)
        noise2: 第二个噪声张量 (1, 3, H, W)
        original_img1: 第一张原始图片 (PIL Image 或 tensor)
        original_img2: 第二张原始图片 (PIL Image 或 tensor)
        em_manager: EDMContextManager 实例
        frac_list: 插值系数列表
        interpolation_type: 插值类型 ('linear' 或 'slerp')
        steps: 采样步数
        sampler: 采样器类型
        sigma_min: 最小噪声水平
        sigma_max: 最大噪声水平
        rho: Karras 噪声调度的幂律参数
        save_path: 保存路径
        title: 图片标题
        image_size: 图像尺寸（默认 256）
    """
    # 确保噪声形状正确
    if noise1.dim() == 4 and noise1.shape[0] > 1:
        noise1 = noise1[0:1]
    if noise2.dim() == 4 and noise2.shape[0] > 1:
        noise2 = noise2[0:1]
    
    # 转换为 PIL Image（如果输入是 tensor）
    if isinstance(original_img1, torch.Tensor):
        original_img1 = tensor_to_image(original_img1)
    if isinstance(original_img2, torch.Tensor):
        original_img2 = tensor_to_image(original_img2)
    
    # 确保原始图片尺寸一致
    target_size = (image_size, image_size)
    if original_img1.size != target_size:
        original_img1 = original_img1.resize(target_size, Image.LANCZOS)
    if original_img2.size != target_size:
        original_img2 = original_img2.resize(target_size, Image.LANCZOS)
    
    # 准备插值结果列表
    interp_images = []
    interp_noises = []
    interp_noise_norms = []
    
    # 添加第一张原始图片和噪声
    interp_images.append(original_img1)
    noise1_pil = tensor_to_image(noise1)
    if noise1_pil.size != target_size:
        noise1_pil = noise1_pil.resize(target_size, Image.LANCZOS)
    interp_noises.append(noise1_pil)
    interp_noise_norms.append(torch.norm(noise1).item())
    
    # 对每个插值系数进行插值和生成
    for frac in frac_list:
        # 进行噪声插值
        if interpolation_type == 'slerp':
            interp_noise = slerp(noise1, noise2, frac)
        else:  # linear
            interp_noise = interpolate_linear(noise1, noise2, frac)
        
        # 使用插值后的噪声生成图像
        interp_img = em_manager._sample_with_noise(
            interp_noise, steps, sampler, sigma_min, sigma_max, rho=rho
        )
        
        # 转换为 PIL Image
        interp_img_pil = tensor_to_image(interp_img)
        if interp_img_pil.size != target_size:
            interp_img_pil = interp_img_pil.resize(target_size, Image.LANCZOS)
        interp_images.append(interp_img_pil)
        
        # 转换插值噪声为 PIL Image
        interp_noise_pil = tensor_to_image(interp_noise)
        if interp_noise_pil.size != target_size:
            interp_noise_pil = interp_noise_pil.resize(target_size, Image.LANCZOS)
        interp_noises.append(interp_noise_pil)
        interp_noise_norms.append(torch.norm(interp_noise).item())
    
    # 添加第二张原始图片和噪声
    interp_images.append(original_img2)
    noise2_pil = tensor_to_image(noise2)
    if noise2_pil.size != target_size:
        noise2_pil = noise2_pil.resize(target_size, Image.LANCZOS)
    interp_noises.append(noise2_pil)
    interp_noise_norms.append(torch.norm(noise2).item())
    
    # 创建图形：2行 x (2 + len(frac_list))列
    num_cols = len(interp_images)
    fig, axes = plt.subplots(2, num_cols, figsize=(5 * num_cols, 10))
    
    # 第一行：插值图片
    for idx, img in enumerate(interp_images):
        axes[0, idx].imshow(img)
        if idx == 0:
            axes[0, idx].set_title('Original 1', fontsize=12)
        elif idx == num_cols - 1:
            axes[0, idx].set_title('Original 2', fontsize=12)
        else:
            frac = frac_list[idx - 1]
            axes[0, idx].set_title(f'Interp {frac:.2f}', fontsize=12)
        axes[0, idx].axis('off')
    
    # 第二行：对应的插值噪声
    for idx, noise in enumerate(interp_noises):
        axes[1, idx].imshow(noise)
        noise_title = 'Noise'
        if idx == 0:
            noise_title = 'Noise 1'
        elif idx == num_cols - 1:
            noise_title = 'Noise 2'
        noise_title += f'\n(norm: {interp_noise_norms[idx]:.2f})'
        axes[1, idx].set_title(noise_title, fontsize=12)
        axes[1, idx].axis('off')
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"插值对比图已保存到: {save_path}")

def add_noise_to_image(image_tensor, sigma, device='cuda'):
    """
    给图像添加指定水平的噪声
    
    Args:
        image_tensor: 输入图像张量，形状为 (1, 3, H, W)，值域 [-1, 1]
        sigma: 噪声水平
        device: 设备
    
    Returns:
        noisy_image: 加噪后的图像张量
    """
    dtype = image_tensor.dtype
    noise = torch.randn_like(image_tensor, device=device, dtype=dtype) * sigma
    noisy_image = image_tensor + noise
    return noisy_image

def create_experiment_folder(base_dir=None):
    """
    创建带时间戳的实验文件夹
    
    Args:
        base_dir: 基础目录，如果为 None 则使用默认路径
    
    Returns:
        实验文件夹路径（绝对路径）
    """
    if base_dir is None:
        # 获取脚本所在目录的父目录（controlnet目录）
        script_dir = os.path.dirname(os.path.abspath(__file__))
        controlnet_dir = os.path.dirname(script_dir)
        base_dir = os.path.join(controlnet_dir, 'sample_results', 'bedroom_edm')
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(base_dir, timestamp)
    os.makedirs(exp_dir, exist_ok=True)
    return exp_dir

def encode_and_reconstruct(image_tensor, em_manager, sigma_start=0.002, sigma_end=80.0):
    """
    对图像进行反转和重建
    
    Args:
        image_tensor: 输入图像张量，形状为 (1, 3, H, W)，值域 [-1, 1]
        em_manager: EDMContextManager 实例
        sigma_start: 反转过程的起始噪声水平（默认 0.002）
        sigma_end: 反转过程的结束噪声水平（默认 80.0）
    
    Returns:
        encoded_noise: 反转后的噪声张量
        reconstructed_image: 重建后的图像张量
        mse: 重建误差
    """
    encoded_noise = em_manager.encode_image(
        image_tensor,
        steps=50,
        solver='heun',
        alpha=1.0,
        rho=7.0,
        sigma_start=sigma_start,
        sigma_end=sigma_end
    )
    
    reconstructed_image = em_manager._sample_with_noise(
        encoded_noise,
        steps=50,
        sampler='heun',
        sigma_min=0.002,  # 生成过程总是到 sigma_min
        sigma_max=sigma_end,
        rho=7.0
    )
    
    mse = torch.mean((image_tensor - reconstructed_image) ** 2).item()
    
    return encoded_noise, reconstructed_image, mse


if __name__ == '__main__':
    # ========== 实验配置参数 ==========
    # 生成和反转的步数配置
    ENCODE_STEPS = 50      # 反转（编码）过程的步数
    DECODE_STEPS = 50      # 生成（解码）过程的步数
    INTERPOLATION_STEPS = 50  # 插值实验中的采样步数

    # ODE 求解器类型：'heun' 或 'euler'
    SOLVER = 'heun'

    # 噪声水平配置
    SIGMA_MIN = 0.002      # 最小噪声水平
    SIGMA_MAX = 80.0       # 最大噪声水平（用于原始噪声生成）
    PERTURBATION_SIGMA = 0.05  # 微扰噪声水平（用于在真实图像上添加噪声的实验）
    
    # 反转过程区间配置
    SIGMA_START = 0.002  # 反转过程的起始噪声水平（默认使用 SIGMA_MIN）
    SIGMA_END = 80.0  # 反转过程的结束噪声水平（默认使用 SIGMA_MAX）

    # 时间步调度配置
    RHO = 7.0              # Karras 噪声调度的幂律参数，控制时间步分布（rho越大，在低噪声区域采样越密集）

    # 反转过程配置
    ENCODE_ALPHA = 1.0     # 反转过程中的 alpha 参数（用于 Heun 求解器）

    # 插值实验配置
    INTERPOLATION_FRAC_LIST = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]  # 插值系数列表，例如 [0.25, 0.5, 0.75] 表示 25%, 50%, 75% 的插值
    INTERPOLATION_TYPE = 'slerp'  # 插值类型: 'linear' 或 'slerp'

    # 数据配置
    NUM_TEST_IMAGES = 2    # 从测试集读取的图片数量
    NUM_GENERATED_IMAGES = 2  # 随机生成的图片数量
    IMAGE_SIZE = 256       # 图像尺寸
    LMDB_PATH = '/home/minchen/bedroom_val_lmdb'  # LMDB 数据库路径

    # ========== 初始化 EDM 模型管理器 ==========
    # 注意：需要将模型路径指向实际的 EDM 模型文件
    model_path = os.path.join(controlnet_dir, 'models', 'edm_bedroom256_ema.pt')
    print("=" * 50)
    print("正在加载 EDM 模型...")
    print("=" * 50)
    EM = cm_edm.EDMContextManager(model_path=model_path, image_size=IMAGE_SIZE, device='cuda')

    # 禁用梯度追踪以加速实验
    torch.set_grad_enabled(False)
    print("已禁用梯度追踪以加速实验")

    # 创建实验文件夹
    exp_dir = create_experiment_folder()
    print(f"实验文件夹: {exp_dir}")

    # ========== 从测试集读取图片并进行反转重建实验 ==========
    print("\n" + "=" * 50)
    print("从测试集读取图片并进行反转重建实验")
    print("=" * 50)

    # 从 LMDB 读取图片
    print(f"正在从 {LMDB_PATH} 读取 {NUM_TEST_IMAGES} 张图片...")
    test_images_pil, image_keys = read_images_from_lmdb(LMDB_PATH, NUM_TEST_IMAGES)

    if len(test_images_pil) < NUM_TEST_IMAGES:
        print(f"警告：只读取到 {len(test_images_pil)} 张图片，少于请求的 {NUM_TEST_IMAGES} 张")

    # 保存真实图片的反转噪声
    real_encoded_noises = []

    # 处理每张真实图片
    for idx, (pil_img, img_key) in enumerate(zip(test_images_pil, image_keys)):
        print(f"\n处理真实图片 {idx + 1} (key: {img_key})...")
        
        # 预处理图片：转换为模型需要的张量格式
        image_tensor = preprocess_image(pil_img, image_size=IMAGE_SIZE, device='cuda')
        
        # 反转和重建
        print("正在进行反转和重建...")
        encoded_noise, reconstructed_image, mse = encode_and_reconstruct(
            image_tensor, EM,
            sigma_start=SIGMA_MIN,
            sigma_end=SIGMA_MAX
        )
        print(f"反转完成，噪声水平: {SIGMA_MIN} -> {SIGMA_MAX}, 重建 MSE: {mse:.6f}")
        
        # 保存反转噪声
        real_encoded_noises.append(encoded_noise)
        
        # 保存对比图：真实图像-反转噪声-生成图像
        comparison_path = os.path.join(exp_dir, f'real_image_{idx+1:02d}_comparison.png')
        visualize_comparison(
            original_img=pil_img,
            noise_img=encoded_noise,
            reconstructed_img=reconstructed_image,
            save_path=comparison_path,
            title=f'Real Image {idx+1} Comparison (MSE: {mse:.6f})',
            image_size=IMAGE_SIZE
        )

    # ========== 在真实图像上添加微扰噪声后再反转的实验 ==========
    print("\n" + "=" * 50)
    print(f"在真实图像上添加微扰噪声 (sigma={PERTURBATION_SIGMA}) 后再反转的实验...")
    print("=" * 50)

    # 保存加噪后真实图片的反转噪声
    noisy_real_encoded_noises = []

    # 处理每张真实图片
    for idx, (pil_img, img_key) in enumerate(zip(test_images_pil, image_keys)):
        print(f"\n处理真实图片 {idx + 1} (key: {img_key})...")
        
        # 预处理图片：转换为模型需要的张量格式
        image_tensor = preprocess_image(pil_img, image_size=IMAGE_SIZE, device='cuda')
        
        # 添加微扰噪声
        print(f"正在添加微扰噪声 (sigma={PERTURBATION_SIGMA})...")
        noisy_image_tensor = add_noise_to_image(image_tensor, PERTURBATION_SIGMA, device='cuda')
        noisy_image_pil = tensor_to_image(noisy_image_tensor)
        
        # 反转和重建
        print("正在进行反转和重建...")
        encoded_noise, reconstructed_image, mse_noisy = encode_and_reconstruct(
            noisy_image_tensor, EM,
            sigma_start=PERTURBATION_SIGMA,
            sigma_end=SIGMA_MAX
        )
        
        # 计算相对于原始图像的重建误差
        mse_original = torch.mean((image_tensor - reconstructed_image) ** 2).item()
        
        print(f"反转完成，噪声水平: {PERTURBATION_SIGMA} -> {SIGMA_MAX}")
        print(f"  相对于加噪图像的重建 MSE: {mse_noisy:.6f}")
        print(f"  相对于原始图像的重建 MSE: {mse_original:.6f}")
        
        # 保存反转噪声
        noisy_real_encoded_noises.append(encoded_noise)
        
        # 保存对比图：原始图像-加噪图像-反转噪声-重建图像
        comparison_path = os.path.join(exp_dir, f'noisy_real_image_{idx+1:02d}_comparison.png')
        title = f'Noisy Real Image {idx+1} Comparison\nMSE vs Noisy: {mse_noisy:.6f}, MSE vs Original: {mse_original:.6f}'
        visualize_comparison_with_noisy(
            original_img=pil_img,
            noisy_img=noisy_image_pil,
            noise_img=encoded_noise,
            reconstructed_img=reconstructed_image,
            save_path=comparison_path,
            title=title,
            perturbation_sigma=PERTURBATION_SIGMA,
            image_size=IMAGE_SIZE
        )

    # ========== 随机采样生成图像并进行反转重建实验 ==========
    print("\n" + "=" * 50)
    print("开始随机采样生成图像...")
    print("=" * 50)

    # 直接生成图片而不保存单独文件
    torch.manual_seed(43)
    shape = (NUM_GENERATED_IMAGES, 3, IMAGE_SIZE, IMAGE_SIZE)
    dtype = getattr(EM.model, 'dtype', torch.float32)
    generator = get_generator("dummy")
    initial_noises = generator.randn(*shape, device='cuda', dtype=dtype) * SIGMA_MAX

    # 打印初始随机噪声的norm
    print("\n初始随机噪声的norm:")
    for i in range(NUM_GENERATED_IMAGES):
        noise = initial_noises[i:i+1]
        noise_norm = torch.norm(noise).item()
        print(f"  噪声 {i+1}: norm = {noise_norm:.2f}")

    # 生成图片
    generated_images = []
    for i in range(NUM_GENERATED_IMAGES):
        noise = initial_noises[i:i+1]
        generated_image = EM._sample_with_noise(
            noise,
            steps=DECODE_STEPS,
            sampler=SOLVER,
            sigma_min=SIGMA_MIN,
            sigma_max=SIGMA_MAX,
            rho=RHO
        )
        generated_images.append(generated_image)
        print(f"已生成图片 {i+1}")

    # 保存生成图片的反转噪声
    generated_encoded_noises = []

    # 对生成的两张图片进行反转重建实验
    print("\n" + "=" * 50)
    print("对生成图像进行反转重建实验...")
    print("=" * 50)

    # 对每张生成的图片进行反转重建
    for idx, generated_image in enumerate(generated_images):
        print(f"\n处理生成图片 {idx + 1}...")
        
        # 反转和重建
        print("正在进行反转和重建...")
        encoded_noise, reconstructed_image, mse = encode_and_reconstruct(
            generated_image, EM,
            sigma_start=SIGMA_MIN,
            sigma_end=SIGMA_MAX
        )
        print(f"反转完成，噪声水平: {SIGMA_MIN} -> {SIGMA_MAX}, 重建 MSE: {mse:.6f}")
        
        # 保存反转噪声
        generated_encoded_noises.append(encoded_noise)
        
        # 保存对比图：生成图像-反转噪声-重建图像
        comparison_path = os.path.join(exp_dir, f'generated_image_{idx+1:02d}_comparison.png')
        visualize_comparison(
            original_img=generated_image,
            noise_img=encoded_noise,
            reconstructed_img=reconstructed_image,
            save_path=comparison_path,
            title=f'Generated Image {idx+1} Comparison (MSE: {mse:.6f})',
            image_size=IMAGE_SIZE
        )

    # ========== 在反转噪声上进行插值实验 ==========
    print("\n" + "=" * 50)
    print("在反转噪声上进行插值实验...")
    print("=" * 50)

    # 0. 在加噪后真实图片的反转噪声之间进行插值
    if len(noisy_real_encoded_noises) >= 2:
        print("\n在加噪后真实图片的反转噪声之间进行插值...")
        noisy_real_noise_interpolation_dir = os.path.join(exp_dir, 'interpolation_noisy_real_encoded_noise')
        os.makedirs(noisy_real_noise_interpolation_dir, exist_ok=True)
        
        # 使用新的可视化函数
        save_path = os.path.join(noisy_real_noise_interpolation_dir, 'interpolation_comparison.png')
        visualize_interpolation(
            noise1=noisy_real_encoded_noises[0],
            noise2=noisy_real_encoded_noises[1],
            original_img1=test_images_pil[0],
            original_img2=test_images_pil[1],
            em_manager=EM,
            frac_list=INTERPOLATION_FRAC_LIST,
            interpolation_type=INTERPOLATION_TYPE,
            steps=INTERPOLATION_STEPS,
            sampler=SOLVER,
            sigma_min=SIGMA_MIN,
            sigma_max=SIGMA_MAX,
            rho=RHO,
            save_path=save_path,
            title='Interpolation: Noisy Real Images Encoded Noise',
            image_size=IMAGE_SIZE
        )

    # 1. 在真实图片的反转噪声之间进行插值
    if len(real_encoded_noises) >= 2:
        print("\n在真实图片的反转噪声之间进行插值...")
        real_noise_interpolation_dir = os.path.join(exp_dir, 'interpolation_real_encoded_noise')
        os.makedirs(real_noise_interpolation_dir, exist_ok=True)
        
        # 使用新的可视化函数
        save_path = os.path.join(real_noise_interpolation_dir, 'interpolation_comparison.png')
        visualize_interpolation(
            noise1=real_encoded_noises[0],
            noise2=real_encoded_noises[1],
            original_img1=test_images_pil[0],
            original_img2=test_images_pil[1],
            em_manager=EM,
            frac_list=INTERPOLATION_FRAC_LIST,
            interpolation_type=INTERPOLATION_TYPE,
            steps=INTERPOLATION_STEPS,
            sampler=SOLVER,
            sigma_min=SIGMA_MIN,
            sigma_max=SIGMA_MAX,
            rho=RHO,
            save_path=save_path,
            title='Interpolation: Real Images Encoded Noise',
            image_size=IMAGE_SIZE
        )

    # 2. 在生成图片的反转噪声之间进行插值
    if len(generated_encoded_noises) >= 2:
        print("\n在生成图片的反转噪声之间进行插值...")
        generated_noise_interpolation_dir = os.path.join(exp_dir, 'interpolation_generated_encoded_noise')
        os.makedirs(generated_noise_interpolation_dir, exist_ok=True)
        
        # 使用新的可视化函数
        save_path = os.path.join(generated_noise_interpolation_dir, 'interpolation_comparison.png')
        visualize_interpolation(
            noise1=generated_encoded_noises[0],
            noise2=generated_encoded_noises[1],
            original_img1=generated_images[0],
            original_img2=generated_images[1],
            em_manager=EM,
            frac_list=INTERPOLATION_FRAC_LIST,
            interpolation_type=INTERPOLATION_TYPE,
            steps=INTERPOLATION_STEPS,
            sampler=SOLVER,
            sigma_min=SIGMA_MIN,
            sigma_max=SIGMA_MAX,
            rho=RHO,
            save_path=save_path,
            title='Interpolation: Generated Images Encoded Noise',
            image_size=IMAGE_SIZE
        )

    # ========== 原始噪声插值实验（保留） ==========
    print("\n" + "=" * 50)
    print("在原始生成噪声上进行插值实验...")
    print("=" * 50)
    noise1 = initial_noises[0:1]  # 第一张图片的噪声
    noise2 = initial_noises[1:2]  # 第二张图片的噪声

    interpolation_dir = os.path.join(exp_dir, 'interpolation_original_noise')
    os.makedirs(interpolation_dir, exist_ok=True)

    # 使用新的可视化函数
    save_path = os.path.join(interpolation_dir, 'interpolation_comparison.png')
    visualize_interpolation(
        noise1=noise1,
        noise2=noise2,
        original_img1=generated_images[0],
        original_img2=generated_images[1],
        em_manager=EM,
        frac_list=INTERPOLATION_FRAC_LIST,
        interpolation_type=INTERPOLATION_TYPE,
        steps=INTERPOLATION_STEPS,
        sampler=SOLVER,
        sigma_min=SIGMA_MIN,
        sigma_max=SIGMA_MAX,
        rho=RHO,
        save_path=save_path,
        title='Interpolation: Original Generated Noise',
        image_size=IMAGE_SIZE
    )

    # ========== 噪声统计特征分析实验 ==========
    print("\n" + "=" * 50)
    print("噪声统计特征分析实验...")
    print("=" * 50)
    print("注意：所有噪声在分析前都会除以 sigma_max 进行归一化")
    print("=" * 50)

    # 准备四种噪声进行分析（每种只分析第一个）
    noise_statistics = []

    # 1. 分析高斯噪声（原始生成的随机噪声）- 只分析第一个
    print("\n分析高斯噪声（原始生成的随机噪声）...")
    if len(initial_noises) > 0:
        gaussian_noise = initial_noises[0:1] / SIGMA_MAX  # 除以 sigma_max
        stats = analyse.analyze_tensor(gaussian_noise, name="Gaussian Noise 1")
        noise_statistics.append(stats)
        print(f"  高斯噪声 1 分析完成")

    # 2. 分析直接反转噪声（从真实图像反转得到的噪声）- 只分析第一个
    print("\n分析直接反转噪声（从真实图像反转得到的噪声）...")
    if len(real_encoded_noises) > 0:
        encoded_noise = real_encoded_noises[0] / SIGMA_MAX  # 除以 sigma_max
        stats = analyse.analyze_tensor(encoded_noise, name="Real Image Encoded Noise 1")
        noise_statistics.append(stats)
        print(f"  真实图像反转噪声 1 分析完成")

    # 3. 分析微扰反转噪声（从添加微扰后的图像反转得到的噪声）- 只分析第一个
    print("\n分析微扰反转噪声（从添加微扰后的图像反转得到的噪声）...")
    if len(noisy_real_encoded_noises) > 0:
        encoded_noise = noisy_real_encoded_noises[0] / SIGMA_MAX  # 除以 sigma_max
        stats = analyse.analyze_tensor(encoded_noise, name="Noisy Real Image Encoded Noise 1")
        noise_statistics.append(stats)
        print(f"  微扰图像反转噪声 1 分析完成")

    # 4. 分析生成图像的反转噪声 - 只分析第一个
    print("\n分析生成图像的反转噪声...")
    if len(generated_encoded_noises) > 0:
        encoded_noise = generated_encoded_noises[0] / SIGMA_MAX  # 除以 sigma_max
        stats = analyse.analyze_tensor(encoded_noise, name="Generated Image Encoded Noise 1")
        noise_statistics.append(stats)
        print(f"  生成图像反转噪声 1 分析完成")

    # 保存统计结果到文件
    stats_output_file = os.path.join(exp_dir, 'noise_statistics.txt')
    with open(stats_output_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("噪声统计特征分析结果\n")
        f.write(f"分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"噪声数量: {len(noise_statistics)}\n")
        f.write(f"注意：所有噪声在分析前都已除以 sigma_max ({SIGMA_MAX}) 进行归一化\n")
        f.write("=" * 80 + "\n\n")
        
        # 按类型分组
        gaussian_stats = [s for s in noise_statistics if 'Gaussian' in s['name']]
        real_encoded_stats = [s for s in noise_statistics if 'Real Image Encoded' in s['name'] and 'Noisy' not in s['name']]
        noisy_encoded_stats = [s for s in noise_statistics if 'Noisy Real Image Encoded' in s['name']]
        generated_encoded_stats = [s for s in noise_statistics if 'Generated Image Encoded' in s['name']]
        
        # 写入每种类型的详细统计
        for stats_list, title in [(gaussian_stats, "高斯噪声（原始生成的随机噪声）"),
                                   (real_encoded_stats, "直接反转噪声（从真实图像反转得到的噪声）"),
                                   (noisy_encoded_stats, "微扰反转噪声（从添加微扰后的图像反转得到的噪声）"),
                                   (generated_encoded_stats, "生成图像的反转噪声")]:
            if not stats_list:
                continue
                
            f.write(f"{title}\n")
            f.write("-" * 80 + "\n")
            
            for stats in stats_list:
                f.write(f"\n{stats['name']}:\n")
                f.write(f"  图像尺寸: {stats['height']} × {stats['width']}\n\n")
                
                f.write("  通道统计:\n")
                for ch in range(3):
                    mean = stats['channel_means'][ch]
                    var = stats['channel_variances'][ch]
                    f.write(f"    通道 {ch}: 均值 = {mean:.6f}, 方差 = {var:.6f}\n")
                
                f.write("\n  通道间相关性:\n")
                f.write(f"    通道0-通道1: {stats['channel_correlation_0_1']:.6f}\n")
                f.write(f"    通道0-通道2: {stats['channel_correlation_0_2']:.6f}\n")
                f.write(f"    通道1-通道2: {stats['channel_correlation_1_2']:.6f}\n")
                
                f.write("\n  空间相关性 (基于通道0):\n")
                f.write(f"    水平方向平均相关性: {stats['horizontal_corr_mean']:.6f}\n")
                f.write(f"    垂直方向平均相关性: {stats['vertical_corr_mean']:.6f}\n")
                f.write("\n")
            
            # 计算该类型的平均值
            if len(stats_list) > 1:
                f.write(f"{title} - 平均值统计:\n")
                f.write("-" * 40 + "\n")
                
                avg_channel_means = np.mean([s['channel_means'] for s in stats_list], axis=0)
                avg_channel_variances = np.mean([s['channel_variances'] for s in stats_list], axis=0)
                avg_corr_01 = np.mean([s['channel_correlation_0_1'] for s in stats_list])
                avg_corr_02 = np.mean([s['channel_correlation_0_2'] for s in stats_list])
                avg_corr_12 = np.mean([s['channel_correlation_1_2'] for s in stats_list])
                avg_horizontal = np.mean([s['horizontal_corr_mean'] for s in stats_list])
                avg_vertical = np.mean([s['vertical_corr_mean'] for s in stats_list])
                
                f.write("  平均通道统计:\n")
                for ch in range(3):
                    f.write(f"    通道 {ch}: 均值 = {avg_channel_means[ch]:.6f}, 方差 = {avg_channel_variances[ch]:.6f}\n")
                
                f.write("\n  平均通道间相关性:\n")
                f.write(f"    通道0-通道1: {avg_corr_01:.6f}\n")
                f.write(f"    通道0-通道2: {avg_corr_02:.6f}\n")
                f.write(f"    通道1-通道2: {avg_corr_12:.6f}\n")
                
                f.write("\n  平均空间相关性:\n")
                f.write(f"    水平方向: {avg_horizontal:.6f}\n")
                f.write(f"    垂直方向: {avg_vertical:.6f}\n")
                f.write("\n")
            
            f.write("=" * 80 + "\n\n")
        
        # 对比总结
        f.write("对比总结:\n")
        f.write("-" * 80 + "\n")
        if gaussian_stats and real_encoded_stats and noisy_encoded_stats and generated_encoded_stats:
            # 计算各类型的平均值
            def calc_avg(stats_list, key):
                values = [s[key] for s in stats_list]
                # 如果值是列表（如channel_means, channel_variances），需要按通道计算平均值
                if isinstance(values[0], (list, np.ndarray)):
                    return np.mean(values, axis=0)  # 返回每个通道的平均值数组
                else:
                    return np.mean(values)  # 返回标量的平均值
            
            f.write("\n通道均值对比:\n")
            gauss_means = calc_avg(gaussian_stats, 'channel_means')
            real_means = calc_avg(real_encoded_stats, 'channel_means')
            noisy_means = calc_avg(noisy_encoded_stats, 'channel_means')
            generated_means = calc_avg(generated_encoded_stats, 'channel_means')
            for ch in range(3):
                f.write(f"  通道 {ch}: 高斯={gauss_means[ch]:.6f}, 直接反转={real_means[ch]:.6f}, "
                        f"微扰反转={noisy_means[ch]:.6f}, 生成反转={generated_means[ch]:.6f}\n")
            
            f.write("\n通道方差对比:\n")
            gauss_vars = calc_avg(gaussian_stats, 'channel_variances')
            real_vars = calc_avg(real_encoded_stats, 'channel_variances')
            noisy_vars = calc_avg(noisy_encoded_stats, 'channel_variances')
            generated_vars = calc_avg(generated_encoded_stats, 'channel_variances')
            for ch in range(3):
                f.write(f"  通道 {ch}: 高斯={gauss_vars[ch]:.6f}, 直接反转={real_vars[ch]:.6f}, "
                        f"微扰反转={noisy_vars[ch]:.6f}, 生成反转={generated_vars[ch]:.6f}\n")
            
            f.write("\n通道间相关性对比:\n")
            f.write(f"  通道0-1: 高斯={calc_avg(gaussian_stats, 'channel_correlation_0_1'):.6f}, "
                    f"直接反转={calc_avg(real_encoded_stats, 'channel_correlation_0_1'):.6f}, "
                    f"微扰反转={calc_avg(noisy_encoded_stats, 'channel_correlation_0_1'):.6f}, "
                    f"生成反转={calc_avg(generated_encoded_stats, 'channel_correlation_0_1'):.6f}\n")
            f.write(f"  通道0-2: 高斯={calc_avg(gaussian_stats, 'channel_correlation_0_2'):.6f}, "
                    f"直接反转={calc_avg(real_encoded_stats, 'channel_correlation_0_2'):.6f}, "
                    f"微扰反转={calc_avg(noisy_encoded_stats, 'channel_correlation_0_2'):.6f}, "
                    f"生成反转={calc_avg(generated_encoded_stats, 'channel_correlation_0_2'):.6f}\n")
            f.write(f"  通道1-2: 高斯={calc_avg(gaussian_stats, 'channel_correlation_1_2'):.6f}, "
                    f"直接反转={calc_avg(real_encoded_stats, 'channel_correlation_1_2'):.6f}, "
                    f"微扰反转={calc_avg(noisy_encoded_stats, 'channel_correlation_1_2'):.6f}, "
                    f"生成反转={calc_avg(generated_encoded_stats, 'channel_correlation_1_2'):.6f}\n")
            
            f.write("\n空间相关性对比:\n")
            f.write(f"  水平方向: 高斯={calc_avg(gaussian_stats, 'horizontal_corr_mean'):.6f}, "
                    f"直接反转={calc_avg(real_encoded_stats, 'horizontal_corr_mean'):.6f}, "
                    f"微扰反转={calc_avg(noisy_encoded_stats, 'horizontal_corr_mean'):.6f}, "
                    f"生成反转={calc_avg(generated_encoded_stats, 'horizontal_corr_mean'):.6f}\n")
            f.write(f"  垂直方向: 高斯={calc_avg(gaussian_stats, 'vertical_corr_mean'):.6f}, "
                    f"直接反转={calc_avg(real_encoded_stats, 'vertical_corr_mean'):.6f}, "
                    f"微扰反转={calc_avg(noisy_encoded_stats, 'vertical_corr_mean'):.6f}, "
                    f"生成反转={calc_avg(generated_encoded_stats, 'vertical_corr_mean'):.6f}\n")

    print(f"\n统计特征分析完成！结果已保存到: {stats_output_file}")

    print("\n" + "=" * 50)
    print("实验完成！")
    print("=" * 50)
