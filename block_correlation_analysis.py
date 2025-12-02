import numpy as np
from scipy.stats import pearsonr

def analyze_block_correlations(input_array):
    """
    分析输入数组中各块的皮尔逊相关系数
    
    参数:
        input_array: 形状为(1, C, height, weight)的numpy数组
        
    返回:
        所有块的前20个最大绝对值相关系数的平均值
    """
    # 确保输入形状正确
    if len(input_array.shape) != 4 or input_array.shape[0] != 1:
        raise ValueError("输入数组形状应为(1, C, height, weight)")
    
    # 移除批次维度
    data = input_array[0]  # 形状变为(C, height, weight)
    C, height, width = data.shape
    
    # 检查图像尺寸是否能被8整除
    if height % 8 != 0 or width % 8 != 0:
        raise ValueError("图像的高度和宽度必须能被8整除")
    
    # 计算块的尺寸
    block_size = 8
    n_blocks_h = height // block_size
    n_blocks_w = width // block_size
    
    all_top_correlations = []
    
    # 对每个通道进行处理
    for c in range(C):
        channel_data = data[c]  # 形状为(height, width)
        
        # 对每个块进行处理
        for i in range(n_blocks_h):
            for j in range(n_blocks_w):
                # 提取当前块
                start_h = i * block_size
                end_h = start_h + block_size
                start_w = j * block_size
                end_w = start_w + block_size
                
                block = channel_data[start_h:end_h, start_w:end_w]
                
                # 将块展平为一维数组
                block_flat = block.flatten()
                n_pixels = len(block_flat)
                
                # 计算所有像素对的皮尔逊相关系数
                correlations = []
                
                # 遍历所有像素对
                for p1 in range(n_pixels):
                    for p2 in range(p1 + 1, n_pixels):
                        # 创建两个像素的"时间序列"
                        # 这里我们使用像素值本身和其邻域值作为序列
                        # 获取像素p1的位置
                        p1_row = p1 // block_size
                        p1_col = p1 % block_size
                        
                        # 获取像素p2的位置
                        p2_row = p2 // block_size
                        p2_col = p2 % block_size
                        
                        # 创建两个序列：每个像素值和其周围3x3邻域的值
                        # 确保两个序列长度相同
                        p1_neighbors = []
                        p2_neighbors = []
                        
                        for di in range(-1, 2):
                            for dj in range(-1, 2):
                                # 检查两个像素的邻域是否都在块内
                                ni1, nj1 = p1_row + di, p1_col + dj
                                ni2, nj2 = p2_row + di, p2_col + dj
                                
                                # 只有当两个邻域都在块内时才添加
                                if (0 <= ni1 < block_size and 0 <= nj1 < block_size and
                                    0 <= ni2 < block_size and 0 <= nj2 < block_size):
                                    p1_neighbors.append(block[ni1, nj1])
                                    p2_neighbors.append(block[ni2, nj2])
                        
                        # 计算皮尔逊相关系数
                        if len(p1_neighbors) > 1:  # 确保有足够的样本点
                            rho, _ = pearsonr(p1_neighbors, p2_neighbors)
                            if not np.isnan(rho):
                                correlations.append(abs(rho))
                
                # 取前20个最大的|ρ|值
                if len(correlations) > 0:
                    correlations.sort(reverse=True)
                    top_20 = correlations[:20]
                    all_top_correlations.extend(top_20)
    
    # 计算所有块的前20个最大绝对值相关系数的平均值
    if len(all_top_correlations) > 0:
        return np.mean(all_top_correlations)
    else:
        return 0.0


# 示例使用
if __name__ == "__main__":
    # 创建一个示例数组 (1, C, height, width)
    C = 3
    height = 32
    width = 32
    
    # 生成随机数据
    np.random.seed(42)
    example_array = np.random.randn(1, C, height, width).astype(np.float32)
    
    print("示例数组形状:", example_array.shape)
    result = analyze_block_correlations(example_array)
    print(f"平均相关系数: {result:.6f}")