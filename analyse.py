import numpy as np
import os
from scipy.stats import pearsonr
from datetime import datetime

def analyze_single_npy(file_path):
    """
    分析单个npy文件并返回统计特征
    
    参数:
        file_path: npy文件路径
        
    返回:
        dict: 包含所有统计特征的字典
    """
    # 加载数据
    data = np.load(file_path)
    
    if data.ndim != 4 or data.shape[0] != 1 or data.shape[1] != 3:
        raise ValueError(f"数据形状应为(1,3,height,width)，但得到的是{data.shape}")
    
    # 去掉批次维度，得到(3, height, width)
    data = data[0]
    
    # 基本形状信息
    height, width = data.shape[1], data.shape[2]
    
    # 1. 均值 - 每个通道的均值
    channel_means = [np.mean(data[i]) for i in range(3)]
    
    # 2. 方差 - 每个通道的方差
    channel_variances = [np.var(data[i]) for i in range(3)]
    
    # 3. 通道间相关性 - 计算每对通道之间的相关性
    channel_correlations = []
    for i in range(3):
        for j in range(i+1, 3):
            corr, _ = pearsonr(data[i].flatten(), data[j].flatten())
            channel_correlations.append(corr)
    
    # 4. 像素间相关性 - 水平和垂直方向的空间相关性
    # 使用第一个通道计算空间相关性
    image_2d = data[0]
    
    # 水平方向像素相关性
    horizontal_corrs = []
    for h in range(height):
        row_pixels = image_2d[h, :]
        if np.std(row_pixels) > 0 and width > 1:
            corr, _ = pearsonr(row_pixels[:-1], row_pixels[1:])
            horizontal_corrs.append(corr)
    
    # 垂直方向像素相关性
    vertical_corrs = []
    for w in range(width):
        col_pixels = image_2d[:, w]
        if np.std(col_pixels) > 0 and height > 1:
            corr, _ = pearsonr(col_pixels[:-1], col_pixels[1:])
            vertical_corrs.append(corr)
    
    # 汇总结果
    result = {
        'filename': os.path.basename(file_path),
        'height': height,
        'width': width,
        'channel_means': channel_means,
        'channel_variances': channel_variances,
        'channel_correlation_0_1': channel_correlations[0] if len(channel_correlations) > 0 else 0,
        'channel_correlation_0_2': channel_correlations[1] if len(channel_correlations) > 1 else 0,
        'channel_correlation_1_2': channel_correlations[2] if len(channel_correlations) > 2 else 0,
        'horizontal_corr_mean': np.mean(horizontal_corrs) if horizontal_corrs else 0,
        'vertical_corr_mean': np.mean(vertical_corrs) if vertical_corrs else 0,
    }
    
    return result

def analyze_folder_npy_files(folder_path):
    """
    遍历文件夹中的所有npy文件并分析
    
    参数:
        folder_path: 文件夹路径
        
    返回:
        list: 所有文件的统计结果列表
    """
    results = []
    
    # 获取文件夹中所有npy文件
    npy_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.npy')]
    
    if not npy_files:
        print(f"在文件夹 '{folder_path}' 中未找到.npy文件")
        return results
    
    print(f"找到 {len(npy_files)} 个.npy文件")
    
    # 分析每个文件
    for i, filename in enumerate(sorted(npy_files), 1):
        file_path = os.path.join(folder_path, filename)
        try:
            print(f"[{i}/{len(npy_files)}] 正在分析: {filename}")
            result = analyze_single_npy(file_path)
            results.append(result)
            print(f"     完成分析")
        except Exception as e:
            print(f"     分析失败: {str(e)}")
            error_result = {
                'filename': filename,
                'error': str(e),
                'height': 0,
                'width': 0,
                'channel_means': [0, 0, 0],
                'channel_variances': [0, 0, 0],
                'channel_correlation_0_1': 0,
                'channel_correlation_0_2': 0,
                'channel_correlation_1_2': 0,
                'horizontal_corr_mean': 0,
                'vertical_corr_mean': 0,
            }
            results.append(error_result)
    
    return results

def save_results_to_txt(folder_path, results):
    """
    将结果保存到result.txt文件中
    
    参数:
        folder_path: 文件夹路径
        results: 分析结果列表
    """
    output_file = os.path.join(folder_path, 'result.txt')
    
    with open(output_file, 'w', encoding='utf-8') as f:
        # 写入标题和时间
        f.write("=" * 80 + "\n")
        f.write(f"NPY文件统计分析结果\n")
        f.write(f"分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"文件数量: {len(results)}\n")
        f.write("=" * 80 + "\n\n")
        
        # 写入每个文件的结果
        for i, result in enumerate(results, 1):
            f.write(f"文件 {i}: {result['filename']}\n")
            f.write("-" * 40 + "\n")
            
            if 'error' in result:
                f.write(f"错误: {result['error']}\n\n")
                continue
            
            f.write(f"图像尺寸: {result['height']} × {result['width']}\n\n")
            
            f.write("通道统计:\n")
            for ch in range(3):
                mean = result['channel_means'][ch]
                var = result['channel_variances'][ch]
                f.write(f"  通道 {ch}: 均值 = {mean:.6f}, 方差 = {var:.6f}\n")
            f.write("\n")
            
            f.write("通道间相关性:\n")
            f.write(f"  通道0-通道1: {result['channel_correlation_0_1']:.6f}\n")
            f.write(f"  通道0-通道2: {result['channel_correlation_0_2']:.6f}\n")
            f.write(f"  通道1-通道2: {result['channel_correlation_1_2']:.6f}\n")
            f.write("\n")
            
            f.write("空间相关性 (基于通道0):\n")
            f.write(f"  水平方向平均相关性: {result['horizontal_corr_mean']:.6f}\n")
            f.write(f"  垂直方向平均相关性: {result['vertical_corr_mean']:.6f}\n")
            f.write("\n" + "=" * 80 + "\n\n")
        
        # 写入汇总统计
        if len(results) > 1 and 'error' not in results[0]:
            f.write("汇总统计 (平均值):\n")
            f.write("-" * 40 + "\n")
            
            # 计算各项的平均值
            valid_results = [r for r in results if 'error' not in r]
            
            if valid_results:
                avg_channel_means = np.mean([r['channel_means'] for r in valid_results], axis=0)
                avg_channel_variances = np.mean([r['channel_variances'] for r in valid_results], axis=0)
                avg_corr_01 = np.mean([r['channel_correlation_0_1'] for r in valid_results])
                avg_corr_02 = np.mean([r['channel_correlation_0_2'] for r in valid_results])
                avg_corr_12 = np.mean([r['channel_correlation_1_2'] for r in valid_results])
                avg_horizontal = np.mean([r['horizontal_corr_mean'] for r in valid_results])
                avg_vertical = np.mean([r['vertical_corr_mean'] for r in valid_results])
                
                f.write("平均通道统计:\n")
                for ch in range(3):
                    f.write(f"  通道 {ch}: 均值 = {avg_channel_means[ch]:.6f}, 方差 = {avg_channel_variances[ch]:.6f}\n")
                f.write("\n")
                
                f.write("平均通道间相关性:\n")
                f.write(f"  通道0-通道1: {avg_corr_01:.6f}\n")
                f.write(f"  通道0-通道2: {avg_corr_02:.6f}\n")
                f.write(f"  通道1-通道2: {avg_corr_12:.6f}\n")
                f.write("\n")
                
                f.write("平均空间相关性:\n")
                f.write(f"  水平方向: {avg_horizontal:.6f}\n")
                f.write(f"  垂直方向: {avg_vertical:.6f}\n")
    
    print(f"\n分析结果已保存到: {output_file}")
    return output_file

def main():
    """
    主函数：遍历文件夹中的npy文件并保存分析结果
    """
    # 设置要分析的文件夹路径
    folder_path = "result/test3"  # 默认当前文件夹，可以修改为其他路径
    
    # 检查文件夹是否存在
    if not os.path.exists(folder_path):
        print(f"错误: 文件夹 '{folder_path}' 不存在")
        return
    
    # 分析所有npy文件
    print(f"开始分析文件夹: {folder_path}")
    results = analyze_folder_npy_files(folder_path)
    
    if results:
        # 保存结果到txt文件
        output_file = save_results_to_txt(folder_path, results)
        
        # 打印简要信息
        print(f"\n分析完成！")
        print(f"共分析 {len(results)} 个文件")
        print(f"结果文件: {output_file}")
        
        # 显示前几个文件的结果
        print(f"\n简要统计 (前3个文件):")
        print("-" * 40)
        for i, result in enumerate(results[:3], 1):
            if 'error' in result:
                print(f"{result['filename']}: 错误 - {result['error']}")
            else:
                print(f"{result['filename']}: {result['height']}×{result['width']}, "
                      f"均值[{result['channel_means'][0]:.3f}, {result['channel_means'][1]:.3f}, {result['channel_means'][2]:.3f}]")
    else:
        print("没有找到可分析的.npy文件")

if __name__ == "__main__":
    main()