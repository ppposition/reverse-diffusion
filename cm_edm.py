"""
EDM Model Manager for NoiseDiffusion experiments
基于 OpenAI Consistency Models 的 EDM 模型进行图像插值实验
"""
import sys
import os

# 添加 consistency_models 到路径
consistency_models_path = '/home/minchen/consistency_models'
if consistency_models_path not in sys.path:
    sys.path.insert(0, consistency_models_path)

import pdb
import shutil
import math
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import yaml

# 导入 consistency_models 的模块
# dist_util 现在已支持 MPI 不可用的情况（会自动回退到单机模式）
from cm import dist_util

try:
    from cm.script_util import create_model_and_diffusion, model_and_diffusion_defaults
    from cm.karras_diffusion import karras_sample, get_sigmas_karras, to_d, sample_heun, sample_euler
    from cm.random_util import get_generator
    from cm.nn import append_zero
except ImportError as e:
    print(f"导入错误: {e}")
    print("请确保 consistency_models 仓库已正确安装")
    raise


def extract_into_tensor(a, t, x_shape):
    """从张量中提取特定索引的值"""
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def interpolate_linear(p0, p1, frac):
    """线性插值"""
    return p0 + (p1 - p0) * frac


@torch.no_grad()
def slerp(p0, p1, fract_mixing: float):
    """
    球面线性插值 (Spherical Linear Interpolation)
    从 lunarring/latentblending 复制
    """
    if p0.dtype == torch.float16:
        recast_to = 'fp16'
    else:
        recast_to = 'fp32'
    
    p0 = p0.double()
    p1 = p1.double()
    norm = torch.linalg.norm(p0) * torch.linalg.norm(p1)
    epsilon = 1e-7
    dot = torch.sum(p0 * p1) / norm
    dot = dot.clamp(-1+epsilon, 1-epsilon)
    
    theta_0 = torch.arccos(dot)
    sin_theta_0 = torch.sin(theta_0)
    theta_t = theta_0 * fract_mixing
    s0 = torch.sin(theta_0 - theta_t) / sin_theta_0
    s1 = torch.sin(theta_t) / sin_theta_0
    interp = p0*s0 + p1*s1
    
    if recast_to == 'fp16':
        interp = interp.half()
    elif recast_to == 'fp32':
        interp = interp.float()
    return interp


class EDMContextManager:
    """
    EDM 模型管理器，用于加载和使用 EDM 模型进行图像插值
    """
    def __init__(self, model_path, image_size=256, device='cuda'):
        """
        初始化 EDM 模型管理器
        
        参数:
            model_path: EDM 模型权重路径 (如 edm_bedroom256_ema.pt)
            image_size: 图像尺寸 (bedroom 和 cat 都是 256)
            device: 设备 ('cuda' 或 'cpu')
        """
        self.device = device
        self.image_size = image_size
        
        # 设置默认参数（基于 LSUN Bedroom/Cat 256）
        # 参考: scripts/launch.sh 中 edm_bedroom256_ema.pt 的配置
        defaults = model_and_diffusion_defaults()
        defaults.update(dict(
            image_size=image_size,
            num_channels=256 if image_size == 256 else 128,
            num_res_blocks=2,
            channel_mult="1,1,2,2,4,4" if image_size == 256 else "1,2,3,4",
            attention_resolutions="32,16,8",
            num_heads=4,
            num_head_channels=64,
            use_fp16=True,  # FlashAttention 需要 fp16，所以必须使用 float16
            class_cond=False,
            resblock_updown=True,  # 关键：bedroom 256 使用 resblock_updown
            use_scale_shift_norm=False,  # 关键：bedroom 256 不使用 scale_shift_norm
            dropout=0.1,  # EDM 训练时使用 0.1 dropout
        ))
        
        # 创建模型和扩散过程
        print(f"正在创建 EDM 模型 (image_size={image_size})...")
        self.model, self.diffusion = create_model_and_diffusion(
            **defaults,
            distillation=False,  # EDM 不是蒸馏模型
        )
        
        # 加载模型权重
        print(f"正在加载模型权重: {model_path}")
        state_dict = dist_util.load_state_dict(model_path, map_location="cpu")
        self.model.load_state_dict(state_dict)
        
        # FlashAttention 需要 fp16，所以必须转换模型
        if defaults.get('use_fp16', False):
            self.model.convert_to_fp16()
        
        self.model.to(device)
        self.model.eval()
        
        print("EDM 模型加载完成！")
    
    @torch.no_grad()
    def encode_image(self, image, target_sigma=None, steps=50, solver='heun', alpha=1.0, rho=7.0, sigma_start=None, sigma_end=None):
        """
        将图像编码到指定噪声水平的噪声空间（反转过程）
        使用 EDM 确定性 ODE 的反向过程求解
        
        参数:
            image: 输入图像张量，形状为 (B, 3, H, W)，值域 [-1, 1]
            target_sigma: 目标噪声水平（已弃用，保留以兼容旧代码，使用 sigma_end 代替）
            steps: 反转步数
            solver: ODE 求解器类型 ('euler' 或 'heun')
            alpha: Heun 方法的参数
            rho: Karras 噪声调度的幂律参数，控制时间步分布
            sigma_start: 反转过程的起始噪声水平（如果为 None，使用 diffusion.sigma_min）
            sigma_end: 反转过程的结束噪声水平（如果为 None，使用 target_sigma 或 diffusion.sigma_max）
        
        返回:
            noisy_image: 编码后的噪声图像
        """
        from cm.karras_diffusion import get_sigmas_karras
        
        # 确定起始和结束的 sigma 水平
        if sigma_start is None:
            sigma_start = self.diffusion.sigma_min
        if sigma_end is None:
            if target_sigma is not None:
                sigma_end = min(target_sigma, self.diffusion.sigma_max)
            else:
                sigma_end = self.diffusion.sigma_max
        
        # 确保 sigma_start < sigma_end
        if sigma_start >= sigma_end:
            raise ValueError(f"sigma_start ({sigma_start}) 必须小于 sigma_end ({sigma_end})")
        
        # 时间步离散化（正向过程：从 sigma_start 到 sigma_end）
        step_indices = torch.arange(steps, dtype=torch.float32, device=self.device)
        t_steps = (sigma_start ** (1 / rho) + step_indices / (steps - 1) * (sigma_end ** (1 / rho) - sigma_start ** (1 / rho))) ** rho
        
        # 确保时间步在模型支持的范围内
        t_steps = torch.clamp(t_steps, min=self.diffusion.sigma_min, max=self.diffusion.sigma_max)
        
        # 主循环（从 sigma_start 到 sigma_end）
        x_next = image.clone().to(torch.float32)
        
        batch_size = image.shape[0]
        
        for i in range(len(t_steps) - 1):
            x_cur = x_next
            t_cur_val = t_steps[i].item() if isinstance(t_steps[i], torch.Tensor) else float(t_steps[i])
            t_next_val = t_steps[i + 1].item() if isinstance(t_steps[i + 1], torch.Tensor) else float(t_steps[i + 1])
            h = t_next_val - t_cur_val
            
            # 将 t_cur 转换为张量（batch_size 维度），denoise 函数需要这个格式
            t_cur_tensor = torch.full((batch_size,), t_cur_val, device=self.device, dtype=torch.float32)
            
            # 计算 ODE 导数: dx/dt = (x - denoise(x, t)) / t
            # 注意：这里使用正向扩散的 ODE（从低噪声到高噪声）
            _, denoised = self.diffusion.denoise(self.model, x_cur, t_cur_tensor)
            d_cur = (x_cur - denoised) / t_cur_val
            
            if solver == 'euler':
                # Euler 方法: x_{i+1} = x_i + h * f(x_i, t_i)
                x_next = x_cur + h * d_cur
            elif solver == 'heun':
                # Heun 方法（二阶）: 使用预测-校正格式
                # 预测步
                x_prime = x_cur + alpha * h * d_cur
                t_prime_val = t_cur_val + alpha * h
                t_prime_tensor = torch.full((batch_size,), t_prime_val, device=self.device, dtype=torch.float32)
                
                # 在预测点评估导数
                _, denoised_prime = self.diffusion.denoise(self.model, x_prime, t_prime_tensor)
                d_prime = (x_prime - denoised_prime) / t_prime_val
                
                # 校正步（二阶更新）
                x_next = x_cur + h * ((1 - 1 / (2 * alpha)) * d_cur + 1 / (2 * alpha) * d_prime)
            else:
                raise ValueError(f"不支持的求解器类型: {solver}，请选择 'euler' 或 'heun'")
        
        return x_next
    
    # @torch.no_grad()
    # def denoise_image(self, noisy_image, sigma, steps=50, sampler='heun'):
    #     """
    #     从噪声图像去噪生成清晰图像
    #     
    #     参数:
    #         noisy_image: 噪声图像
    #         sigma: 起始噪声水平
    #         steps: 采样步数
    #         sampler: 采样器类型 ('heun', 'euler', 'dpm' 等)
    #     
    #     返回:
    #         denoised_image: 去噪后的图像
    #     """
    #     # 创建噪声调度
    #     sigmas = get_sigmas_karras(
    #         steps, 
    #         sigma_min=self.diffusion.sigma_min,
    #         sigma_max=sigma,
    #         rho=7.0,
    #         device=self.device
    #     )
    #     
    #     # 定义去噪函数
    #     def denoiser(x_t, sigma_t):
    #         _, denoised = self.diffusion.denoise(self.model, x_t, sigma_t)
    #         return denoised.clamp(-1, 1)
    #     
    #     # 使用采样器去噪
    #     generator = get_generator("dummy")
    #     if sampler == 'heun':
    #         x_0 = sample_heun(denoiser, noisy_image, sigmas, generator)
    #     elif sampler == 'euler':
    #         x_0 = sample_euler(denoiser, noisy_image, sigmas, generator)
    #     else:
    #         # 默认使用 heun
    #         x_0 = sample_heun(denoiser, noisy_image, sigmas, generator)
    #     
    #     return x_0.clamp(-1, 1)
    
    @torch.no_grad()
    def sample_images(self, num_samples=4, batch_size=1, steps=50, 
                     sampler='heun', sigma_min=0.002, sigma_max=80.0, 
                     out_dir='samples', seed=42, return_noise=False):
        """
        随机采样生成图像
        
        参数:
            num_samples: 要生成的图像数量
            batch_size: 批次大小
            steps: 采样步数
            sampler: 采样器类型 ('heun', 'euler', 'dpm' 等)
            sigma_min: 最小噪声水平
            sigma_max: 最大噪声水平
            out_dir: 输出目录
            seed: 随机种子
            return_noise: 是否返回初始噪声（用于插值）
        
        返回:
            如果 return_noise=True，返回 noises 张量
            否则返回 None
        """
        torch.manual_seed(seed)
        
        # 创建输出目录
        os.makedirs(out_dir, exist_ok=True)
        
        print(f"正在生成 {num_samples} 张随机图像...")
        
        all_noises = []
        generated = 0
        
        while generated < num_samples:
            current_batch_size = min(batch_size, num_samples - generated)
            
            # 使用 karras_sample 进行采样
            shape = (current_batch_size, 3, self.image_size, self.image_size)
            
            # 生成初始噪声
            dtype = getattr(self.model, 'dtype', torch.float32)
            generator = get_generator("dummy")
            initial_noise = generator.randn(*shape, device=self.device, dtype=dtype) * sigma_max
            
            if return_noise:
                all_noises.append(initial_noise.clone())
            
            # 使用指定的初始噪声进行采样
            samples = self._sample_with_noise(
                initial_noise,
                steps=steps,
                sampler=sampler,
                sigma_min=sigma_min,
                sigma_max=sigma_max,
            )
            
            # 转换为图像并保存
            samples = ((samples + 1) * 127.5).clamp(0, 255).to(torch.uint8)
            samples = samples.permute(0, 2, 3, 1).cpu().numpy()
            
            for i, img_array in enumerate(samples):
                img = Image.fromarray(img_array)
                img_path = os.path.join(out_dir, f'{generated:03d}.png')
                img.save(img_path)
                print(f"已保存: {img_path}")
                generated += 1
        
        print(f"采样完成！共生成 {num_samples} 张图像")
        
        if return_noise:
            return torch.cat(all_noises, dim=0) if all_noises else None
        return None
    
    @torch.no_grad()
    def _sample_with_noise(self, initial_noise, steps=50, sampler='heun',
                          sigma_min=0.002, sigma_max=80.0, rho=7.0):
        """
        使用指定的初始噪声进行采样
        
        参数:
            initial_noise: 初始噪声张量
            steps: 采样步数
            sampler: 采样器类型
            sigma_min: 最小噪声水平
            sigma_max: 最大噪声水平
            rho: Karras 噪声调度的幂律参数，控制时间步分布
        
        返回:
            生成的图像张量
        """
        from cm.karras_diffusion import get_sigmas_karras, sample_heun, sample_euler
        from cm.random_util import get_generator
        
        sigmas = get_sigmas_karras(steps, sigma_min, sigma_max, rho=rho, device=self.device)
        
        def denoiser(x_t, sigma):
            _, denoised = self.diffusion.denoise(self.model, x_t, sigma)
            return denoised.clamp(-1, 1)
        
        generator = get_generator("dummy")
        x_T = initial_noise
        
        if sampler == 'heun':
            from cm.karras_diffusion import sample_heun
            x_0 = sample_heun(denoiser, x_T, sigmas, generator)
        elif sampler == 'euler':
            from cm.karras_diffusion import sample_euler
            x_0 = sample_euler(denoiser, x_T, sigmas, generator)
        else:
            x_0 = sample_heun(denoiser, x_T, sigmas, generator)
        
        return x_0.clamp(-1, 1)
    
    @torch.no_grad()
    def interpolate_noise(self, noise1, noise2, frac_list=[0.25, 0.5, 0.75],
                         interpolation_type='linear', steps=50, sampler='heun',
                         sigma_min=0.002, sigma_max=80.0, rho=7.0, out_dir='interpolation'):
        """
        对两组噪声进行插值并生成图像
        
        参数:
            noise1: 第一组噪声张量，形状为 (1, 3, H, W)
            noise2: 第二组噪声张量，形状为 (1, 3, H, W)
            frac_list: 插值系数列表，例如 [0.25, 0.5, 0.75] 表示 25%, 50%, 75% 的插值
            interpolation_type: 插值类型 ('linear' 或 'slerp')
            steps: 采样步数
            sampler: 采样器类型
            sigma_min: 最小噪声水平
            sigma_max: 最大噪声水平
            rho: Karras 噪声调度的幂律参数，控制时间步分布
            out_dir: 输出目录
        """
        os.makedirs(out_dir, exist_ok=True)
        
        print(f"正在对噪声进行 {interpolation_type} 插值...")
        
        # 确保噪声形状正确
        if noise1.dim() == 4 and noise1.shape[0] > 1:
            noise1 = noise1[0:1]
        if noise2.dim() == 4 and noise2.shape[0] > 1:
            noise2 = noise2[0:1]
        
        # 保存原始图像
        img1 = self._sample_with_noise(noise1, steps, sampler, sigma_min, sigma_max, rho=rho)
        img2 = self._sample_with_noise(noise2, steps, sampler, sigma_min, sigma_max, rho=rho)
        
        img1_pil = Image.fromarray(((img1[0].permute(1, 2, 0) + 1) * 127.5).clamp(0, 255).cpu().numpy().astype(np.uint8))
        img2_pil = Image.fromarray(((img2[0].permute(1, 2, 0) + 1) * 127.5).clamp(0, 255).cpu().numpy().astype(np.uint8))
        img1_pil.save(os.path.join(out_dir, '000_original1.png'))
        img2_pil.save(os.path.join(out_dir, f'{len(frac_list)+1:03d}_original2.png'))
        print(f"已保存原始图像: {out_dir}/000_original1.png 和 {out_dir}/{len(frac_list)+1:03d}_original2.png")
        
        # 对每个插值系数进行插值和生成
        for idx, frac in enumerate(frac_list):
            print(f"正在生成插值图像 {idx+1}/{len(frac_list)} (frac={frac:.2f})...")
            
            # 进行噪声插值
            if interpolation_type == 'slerp':
                interp_noise = slerp(noise1, noise2, frac)
            else:  # linear
                interp_noise = interpolate_linear(noise1, noise2, frac)
            
            # 使用插值后的噪声生成图像
            interp_img = self._sample_with_noise(interp_noise, steps, sampler, sigma_min, sigma_max, rho=rho)
            
            # 保存图像
            img_array = ((interp_img[0].permute(1, 2, 0) + 1) * 127.5).clamp(0, 255).cpu().numpy().astype(np.uint8)
            img_pil = Image.fromarray(img_array)
            img_path = os.path.join(out_dir, f'{idx+1:03d}_frac_{frac:.2f}.png')
            img_pil.save(img_path)
            print(f"已保存: {img_path}")
        
        print(f"插值完成！共生成 {len(frac_list)} 张插值图像")
    
    # @torch.no_grad()
    # def interpolate_new(self, img1, img2, interpolation_type="noisediffusion", 
    #                    target_sigma=40.0, encode_steps=50, decode_steps=50,
    #                    frac_list=[0.17, 0.33, 0.5, 0.67, 0.83],
    #                    name_list=[1, 3, 5, 7, 9],
    #                    out_dir='blend', sampler='heun'):
    #     """
    #     在两个图像之间进行插值
    #     
    #     参数:
    #         img1: 第一张图像 (PIL Image 或 torch.Tensor)
    #         img2: 第二张图像 (PIL Image 或 torch.Tensor)
    #         interpolation_type: 插值类型 ("slerp", "noise", "noisediffusion")
    #         target_sigma: 目标噪声水平（用于反转）
    #         encode_steps: 编码（反转）步数
    #         decode_steps: 解码（采样）步数
    #         frac_list: 插值系数列表
    #         name_list: 输出文件名编号列表
    #         out_dir: 输出目录
    #         sampler: 采样器类型
    #     """
    #     torch.manual_seed(49)
    #     
    #     # 创建输出目录
    #     shutil.rmtree(out_dir, ignore_errors=True)
    #     os.makedirs(out_dir, exist_ok=True)
    #     
    #     # 预处理图像
    #     if isinstance(img1, Image.Image):
    #         img1.save(f'{out_dir}/{0:03d}.png')
    #         img2.save(f'{out_dir}/{2:03d}.png')
    #         if img1.mode == 'RGBA':
    #             img1 = img1.convert('RGB')
    #         if img2.mode == 'RGBA':
    #             img2 = img2.convert('RGB')
    #         
    #         # 转换为张量并归一化到 [-1, 1]
    #         img1 = torch.tensor(np.array(img1)).permute(2, 0, 1).unsqueeze(0).float()
    #         img2 = torch.tensor(np.array(img2)).permute(2, 0, 1).unsqueeze(0).float()
    #         img1 = (img1 / 127.5 - 1.0).to(self.device)
    #         img2 = (img2 / 127.5 - 1.0).to(self.device)
    #     
    #     # 调整图像大小
    #     if img1.shape[-1] != self.image_size:
    #         img1 = F.interpolate(img1, size=(self.image_size, self.image_size), mode='bilinear', align_corners=False)
    #     if img2.shape[-1] != self.image_size:
    #         img2 = F.interpolate(img2, size=(self.image_size, self.image_size), mode='bilinear', align_corners=False)
    #     
    #     # 将图像编码到噪声空间（反转过程）
    #     print(f"正在编码图像1到噪声空间 (sigma={target_sigma})...")
    #     n1 = self.encode_image(img1, target_sigma, steps=encode_steps)
    #     
    #     print(f"正在编码图像2到噪声空间 (sigma={target_sigma})...")
    #     n2 = self.encode_image(img2, target_sigma, steps=encode_steps)
    #     
    #     # 保存参数
    #     kwargs = dict(
    #         interpolation_type=interpolation_type,
    #         target_sigma=target_sigma,
    #         encode_steps=encode_steps,
    #         decode_steps=decode_steps,
    #         sampler=sampler,
    #     )
    #     yaml.dump(kwargs, open(f'{out_dir}/args.yaml', 'w'))
    #     
    #     # 对每个插值系数进行插值和生成
    #     for num in range(len(frac_list)):
    #         frac = frac_list[num]
    #         name = name_list[num]
    #         
    #         print(f"正在生成插值图像 {name} (frac={frac:.2f})...")
    #         
    #         # 根据插值类型进行插值
    #         if interpolation_type == "slerp":
    #             noisy_interp = slerp(n1, n2, frac)
    #             
    #         elif interpolation_type == "noise":
    #             # 在噪声空间进行线性插值
    #             noisy_interp = interpolate_linear(n1, n2, frac)
    #             
    #         elif interpolation_type == "noisediffusion":
    #             # NoiseDiffusion 插值方法
    #             coef = 2.0
    #             gamma = 0
    #             alpha = math.cos(math.radians(frac * 90))
    #             beta = math.sin(math.radians(frac * 90))
    #             l = alpha / beta
    #             
    #             alpha = ((1 - gamma * gamma) * l * l / (l * l + 1))**0.5
    #             beta = ((1 - gamma * gamma) / (l * l + 1))**0.5
    #             
    #             mu = 1.2 * alpha / (alpha + beta)
    #             nu = 1.2 * beta / (alpha + beta)
    #             
    #             n1_clipped = torch.clamp(n1, -coef, coef)
    #             n2_clipped = torch.clamp(n2, -coef, coef)
    #             
    #             # NoiseDiffusion 插值：在噪声空间和原始图像空间之间进行插值
    #             # 使用目标噪声水平作为参考
    #             noise = torch.randn_like(n1)
    #             
    #             # 计算噪声权重（基于目标噪声水平）
    #             # 这里我们使用一个简化的方法，假设在目标噪声水平下进行插值
    #             sigma_ratio = 1.0  # 在目标噪声水平下
    #             
    #             noisy_interp = (
    #                 alpha * n1_clipped + beta * n2_clipped +
    #                 (mu - alpha) * img1 * (1 - sigma_ratio) +
    #                 (nu - beta) * img2 * (1 - sigma_ratio) +
    #                 gamma * noise * sigma_ratio
    #             )
    #             noisy_interp = torch.clamp(noisy_interp, -coef, coef)
    #         else:
    #             raise ValueError(f"未知的插值类型: {interpolation_type}")
    #         
    #         # 从噪声空间解码生成图像
    #         denoised = self.denoise_image(
    #             noisy_interp, 
    #             sigma=target_sigma, 
    #             steps=decode_steps,
    #             sampler=sampler
    #         )
    #         
    #         # 后处理：转换为 PIL Image 并保存
    #         image = (denoised.permute(0, 2, 3, 1) * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)
    #         Image.fromarray(image[0]).save(f'{out_dir}/{name:03d}.png')
    #         
    #         print(f"已保存: {out_dir}/{name:03d}.png")

