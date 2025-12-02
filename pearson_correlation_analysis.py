import numpy as np
from scipy.stats import pearsonr
import os

def load_npy_file(file_path):
    """
    加载npy文件
    
    参数:
        file_path (str): npy文件路径
        
    返回:
        numpy.ndarray: 加载的数组
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件 {file_path} 不存在")
    
    return np.load(file_path)

def divide_into_blocks(array, block_size=8):
    """
    将数组划分为C×8×8的块
    
    参数:
        array (numpy.ndarray): 输入数组，形状为 (N, C, H, W) 或 (C, H, W) 或 (H, W, C)
        block_size (int): 块的大小，默认为8
        
    返回:
        list: 包含所有块的列表，每个块形状为 (C, block_size, block_size)
    """
    # 处理不同维度的数组
    if len(array.shape) == 4:  # (N, C, H, W) 格式
        # 如果是4D数组，只处理第一个样本
        array = array[0]
        print(f"检测到4D数组，使用第一个样本进行分析")
    
    # 确保数组是 (C, H, W) 格式
    if len(array.shape) == 3:
        # 对于(1,3,32,32)格式，取第一个样本后应该是(3,32,32)
        # 我们需要判断这是(C,H,W)还是(H,W,C)格式
        # 通常通道数C会比较小（如3或1），而H和W会比较大
        # 如果第一个维度很小（<=4），且后两个维度相等且较大，很可能是(C,H,W)格式
        if array.shape[0] <= 4 and array.shape[1] == array.shape[2] and array.shape[1] > 10:
            # 很可能是(C,H,W)格式，不需要转置
            print(f"检测到 (C, H, W) 格式数组，无需转置")
        else:
            # 可能是(H,W,C)格式，需要转置
            array = np.transpose(array, (2, 0, 1))  # 转换为 (C, H, W)
            print(f"数组从 (H, W, C) 格式转换为 (C, H, W) 格式")
    
    C, H, W = array.shape
    print(f"处理后的数组形状: (C={C}, H={H}, W={W})")
    
    # 检查图像尺寸是否能被块大小整除
    if H % block_size != 0 or W % block_size != 0:
        raise ValueError(f"图像尺寸 ({H}, {W}) 不能被块大小 {block_size} 整除")
    
    blocks = []
    
    # 计算块的数量
    num_blocks_h = H // block_size
    num_blocks_w = W // block_size
    
    # 划分块
    for i in range(num_blocks_h):
        for j in range(num_blocks_w):
            h_start = i * block_size
            h_end = h_start + block_size
            w_start = j * block_size
            w_end = w_start + block_size
            
            block = array[:, h_start:h_end, w_start:w_end]
            blocks.append(block)
    
    return blocks

def calculate_pearson_correlation(block, top_n=20):
    """
    计算块内所有像素对的皮尔逊相关系数，并取前top_n个最大的|ρ|值
    
    参数:
        block (numpy.ndarray): 输入块，形状为 (C, block_size, block_size)
        top_n (int): 要取的最大绝对值相关系数的数量，默认为20
        
    返回:
        dict: 包含相关系数统计信息的字典
    """
    C, H, W = block.shape
    
    # 将块展平为 (C, H*W) 形状
    flattened = block.reshape(C, H * W)
    
    # 计算所有像素对之间的皮尔逊相关系数
    correlations = []
    
    # 对于每个通道，计算所有像素对之间的相关系数
    for c in range(C):
        channel_pixels = flattened[c]
        total_pixels = len(channel_pixels)
        
        # 计算该通道内所有像素对之间的相关系数
        # 为了避免计算量过大，我们随机采样一部分像素对
        max_pairs = min(1000, total_pixels * (total_pixels - 1) // 2)  # 限制最大计算量
        
        if total_pixels > 1:
            # 生成随机像素对索引
            pair_indices = []
            while len(pair_indices) < max_pairs:
                i = np.random.randint(0, total_pixels)
                j = np.random.randint(0, total_pixels)
                if i != j and (i, j) not in pair_indices and (j, i) not in pair_indices:
                    pair_indices.append((i, j))
            
            # 计算这些像素对的相关系数
            for i, j in pair_indices:
                # 创建两个像素的"数组"，包含它们周围的局部上下文
                # 这里我们简单地使用单个像素值，但可以考虑使用局部区域
                pixel_i = np.array([channel_pixels[i]])
                pixel_j = np.array([channel_pixels[j]])
                
                # 计算相关系数（对于单个值，这实际上是在计算它们的关系）
                # 为了使计算有意义，我们需要考虑像素的上下文信息
                # 让我们使用像素的位置信息作为额外的维度
                pos_i = np.array([i % W, i // W])  # 像素i的(x,y)坐标
                pos_j = np.array([j % W, j // W])  # 像素j的(x,y)坐标
                
                # 将像素值和位置信息组合
                feature_i = np.array([channel_pixels[i], pos_i[0], pos_i[1]])
                feature_j = np.array([channel_pixels[j], pos_j[0], pos_j[1]])
                
                try:
                    corr, _ = pearsonr(feature_i, feature_j)
                    if not np.isnan(corr):
                        correlations.append(corr)
                except:
                    # 如果计算失败，跳过这对像素
                    pass
    
    # 计算通道间的相关性
    for c1 in range(C):
        for c2 in range(c1+1, C):
            corr, _ = pearsonr(flattened[c1], flattened[c2])
            if not np.isnan(corr):
                correlations.append(corr)
    
    # 计算绝对值并排序
    abs_correlations = [abs(corr) for corr in correlations]
    
    # 取前top_n个最大的绝对值相关系数
    if len(abs_correlations) > top_n:
        # 获取前top_n个最大值的索引
        top_indices = np.argsort(abs_correlations)[-top_n:]
        top_correlations = [correlations[i] for i in top_indices]
    else:
        top_correlations = correlations
    
    # 计算统计信息
    result = {
        'mean_correlation': np.mean(correlations) if correlations else 0,
        'max_correlation': np.max(correlations) if correlations else 0,
        'min_correlation': np.min(correlations) if correlations else 0,
        'std_correlation': np.std(correlations) if correlations else 0,
        'correlation_count': len(correlations),
        'correlations': correlations,
        'top_n_abs_correlations': top_correlations,
        'mean_top_n_abs_correlation': np.mean([abs(c) for c in top_correlations]) if top_correlations else 0
    }
    
    return result

def analyze_npy_file(file_path, block_size=8):
    """
    分析npy文件中的数组，计算每个块的皮尔逊相关系数
    
    参数:
        file_path (str): npy文件路径
        block_size (int): 块的大小，默认为8
        
    返回:
        list: 包含所有块分析结果的列表
    """
    # 加载数组
    array = load_npy_file(file_path)
    print(f"加载的数组形状: {array.shape}")
    
    # 如果是4D数组，提供额外信息
    if len(array.shape) == 4:
        N, C, H, W = array.shape
        print(f"检测到4D数组: 批次大小={N}, 通道数={C}, 高度={H}, 宽度={W}")
        print(f"将只分析第一个样本")
    
    # 划分为块
    blocks = divide_into_blocks(array, block_size)
    print(f"划分为 {len(blocks)} 个 {block_size}×{block_size} 的块")
    
    # 分析每个块
    results = []
    for i, block in enumerate(blocks):
        result = calculate_pearson_correlation(block)
        result['block_index'] = i
        results.append(result)
        
        if i < 5:  # 只打印前5个块的信息
            print(f"块 {i}: 平均相关系数 = {result['mean_correlation']:.4f}, "
                  f"最大相关系数 = {result['max_correlation']:.4f}, "
                  f"最小相关系数 = {result['min_correlation']:.4f}")
    
    # 计算整体统计信息
    all_correlations = []
    for result in results:
        all_correlations.extend(result['correlations'])
    
    overall_stats = {
        'overall_mean': np.mean(all_correlations) if all_correlations else 0,
        'overall_max': np.max(all_correlations) if all_correlations else 0,
        'overall_min': np.min(all_correlations) if all_correlations else 0,
        'overall_std': np.std(all_correlations) if all_correlations else 0,
        'total_correlations': len(all_correlations)
    }
    
    print(f"\n整体统计信息:")
    print(f"平均相关系数: {overall_stats['overall_mean']:.4f}")
    print(f"最大相关系数: {overall_stats['overall_max']:.4f}")
    print(f"最小相关系数: {overall_stats['overall_min']:.4f}")
    print(f"相关系数标准差: {overall_stats['overall_std']:.4f}")
    print(f"总相关系数对数: {overall_stats['total_correlations']}")
    
    return results, overall_stats

def save_results(results, overall_stats, output_path):
    """
    保存分析结果到文件
    
    参数:
        results (list): 每个块的分析结果
        overall_stats (dict): 整体统计信息
        output_path (str): 输出文件路径
    """
    with open(output_path, 'w') as f:
        f.write("皮尔逊相关系数分析结果\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("整体统计信息:\n")
        f.write(f"平均相关系数: {overall_stats['overall_mean']:.4f}\n")
        f.write(f"最大相关系数: {overall_stats['overall_max']:.4f}\n")
        f.write(f"最小相关系数: {overall_stats['overall_min']:.4f}\n")
        f.write(f"相关系数标准差: {overall_stats['overall_std']:.4f}\n")
        f.write(f"总相关系数对数: {overall_stats['total_correlations']}\n\n")
        
        f.write("各块详细结果:\n")
        f.write("-" * 50 + "\n")
        
        for result in results:
            f.write(f"块 {result['block_index']}:\n")
            f.write(f"  平均相关系数: {result['mean_correlation']:.4f}\n")
            f.write(f"  最大相关系数: {result['max_correlation']:.4f}\n")
            f.write(f"  最小相关系数: {result['min_correlation']:.4f}\n")
            f.write(f"  相关系数标准差: {result['std_correlation']:.4f}\n")
            f.write(f"  相关系数对数: {result['correlation_count']}\n")
            f.write("\n")
    
    print(f"结果已保存到: {output_path}")

if __name__ == "__main__":
    # 示例用法
    import argparse
    
    parser = argparse.ArgumentParser(description='分析npy文件中数组的皮尔逊相关系数')
    parser.add_argument('input_file', type=str, help='输入的npy文件路径')
    parser.add_argument('--block_size', type=int, default=8, help='块的大小，默认为8')
    parser.add_argument('--output', type=str, default='correlation_analysis_results.txt', 
                       help='输出文件路径，默认为correlation_analysis_results.txt')
    
    args = parser.parse_args()
    
    try:
        results, overall_stats = analyze_npy_file(args.input_file, args.block_size)
        save_results(results, overall_stats, args.output)
    except Exception as e:
        print(f"错误: {e}")