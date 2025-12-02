"""Simple noise analysis tool: directly import saved noise images and original images for analysis"""

import numpy as np
import torch
import PIL.Image
import matplotlib.pyplot as plt
from scipy import stats
import os

def load_image(image_path):
    """Load image and convert to numpy array"""
    img = PIL.Image.open(image_path).convert('RGB')
    return np.array(img)

def analyze_noise_vs_gaussian(noise_path, original_path=None, save_dir='analysis_results'):
    """Analyze similarity between noise image and Gaussian noise"""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Load noise image
    noise_img = np.load(noise_path)
    
    # Normalize to [0, 1] range
    noise_normalized = noise_img.astype(np.float32) / 255.0
    # Generate pure Gaussian noise with same scale
    noise_mean = np.mean(noise_normalized)
    noise_std = np.std(noise_normalized)
    pure_gaussian = np.random.normal(noise_mean, noise_std, noise_normalized.shape)
    
    # Flatten arrays for statistical analysis
    noise_flat = noise_normalized.flatten()
    gaussian_flat = pure_gaussian.flatten()
    
    # Create analysis charts
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Display noise image
    axes[0, 0].imshow(noise_img[0,0])
    axes[0, 0].set_title('Diffusion Noise Image')
    axes[0, 0].axis('off')
    
    # 2. Display pure Gaussian noise
    axes[0, 1].imshow((pure_gaussian[0,0] * 255).clip(0, 255).astype(np.uint8))
    axes[0, 1].set_title('Pure Gaussian Noise')
    axes[0, 1].axis('off')
    
    # 3. Histogram comparison
    axes[1, 0].hist(noise_flat, bins=50, alpha=0.7, label='Diffusion Noise', density=True)
    axes[1, 0].hist(gaussian_flat, bins=50, alpha=0.7, label='Pure Gaussian Noise', density=True)
    axes[1, 0].set_title('Noise Distribution Comparison')
    axes[1, 0].legend()
    axes[1, 0].set_xlabel('Pixel Value')
    axes[1, 0].set_ylabel('Density')
    
    # 4. Q-Q plot
    stats.probplot(noise_flat, dist="norm", plot=axes[1, 1])
    axes[1, 1].set_title('Diffusion Noise Q-Q Plot')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'noise_comparison.png'), dpi=300)
    plt.close()
    
    # Calculate statistics
    noise_stats = {
        'mean': np.mean(noise_flat),
        'std': np.std(noise_flat),
        'skew': stats.skew(noise_flat),
        'kurtosis': stats.kurtosis(noise_flat)
    }
    
    gaussian_stats = {
        'mean': np.mean(gaussian_flat),
        'std': np.std(gaussian_flat),
        'skew': stats.skew(gaussian_flat),
        'kurtosis': stats.kurtosis(gaussian_flat)
    }
    
    # Kolmogorov-Smirnov test
    ks_statistic, ks_p_value = stats.ks_2samp(noise_flat, gaussian_flat)
    
    # Print results
    print("Noise Statistical Analysis Results:")
    print("===================================")
    print(f"Diffusion Noise Statistics:")
    print(f"  Mean={noise_stats['mean']:.4f}")
    print(f"  Std={noise_stats['std']:.4f}")
    print(f"  Skewness={noise_stats['skew']:.4f}")
    print(f"  Kurtosis={noise_stats['kurtosis']:.4f}")
    print(f"\nPure Gaussian Noise Statistics:")
    print(f"  Mean={gaussian_stats['mean']:.4f}")
    print(f"  Std={gaussian_stats['std']:.4f}")
    print(f"  Skewness={gaussian_stats['skew']:.4f}")
    print(f"  Kurtosis={gaussian_stats['kurtosis']:.4f}")
    print(f"\nKolmogorov-Smirnov Test:")
    print(f"  Statistic={ks_statistic:.4f}, p-value={ks_p_value:.4f}")
    
    if ks_p_value > 0.05:
        print("\nConclusion: Cannot reject the hypothesis that the two distributions are the same (p > 0.05)")
        print("Diffusion noise and pure Gaussian noise are not significantly different statistically")
    else:
        print("\nConclusion: The two distributions are significantly different (p <= 0.05)")
        print("Diffusion noise and pure Gaussian noise are significantly different statistically")
    
    # Save results to file
    with open(os.path.join(save_dir, 'analysis_results.txt'), 'w') as f:
        f.write("Noise Statistical Analysis Results\n")
        f.write("===================================\n\n")
        f.write(f"Diffusion Noise Statistics:\n")
        f.write(f"  Mean={noise_stats['mean']:.4f}\n")
        f.write(f"  Std={noise_stats['std']:.4f}\n")
        f.write(f"  Skewness={noise_stats['skew']:.4f}\n")
        f.write(f"  Kurtosis={noise_stats['kurtosis']:.4f}\n\n")
        f.write(f"Pure Gaussian Noise Statistics:\n")
        f.write(f"  Mean={gaussian_stats['mean']:.4f}\n")
        f.write(f"  Std={gaussian_stats['std']:.4f}\n")
        f.write(f"  Skewness={gaussian_stats['skew']:.4f}\n")
        f.write(f"  Kurtosis={gaussian_stats['kurtosis']:.4f}\n\n")
        f.write(f"Kolmogorov-Smirnov Test:\n")
        f.write(f"  Statistic={ks_statistic:.4f}\n")
        f.write(f"  p-value={ks_p_value:.4f}\n\n")
        
        if ks_p_value > 0.05:
            f.write("Conclusion: Cannot reject the hypothesis that the two distributions are the same (p > 0.05)\n")
            f.write("Diffusion noise and pure Gaussian noise are not significantly different statistically\n")
        else:
            f.write("Conclusion: The two distributions are significantly different (p <= 0.05)\n")
            f.write("Diffusion noise and pure Gaussian noise are significantly different statistically\n")
    
    return {
        'noise_stats': noise_stats,
        'gaussian_stats': gaussian_stats,
        'ks_statistic': ks_statistic,
        'ks_p_value': ks_p_value
    }

def compare_two_noises(noise1_path, noise2_path, save_dir='analysis_results'):
    """Compare similarity between two noise images"""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Load two noise images
    noise1_img = load_image(noise1_path)
    noise2_img = load_image(noise2_path)
    
    # Normalize to [0, 1] range
    noise1_normalized = noise1_img.astype(np.float32) / 255.0
    noise2_normalized = noise2_img.astype(np.float32) / 255.0
    
    # Flatten arrays for statistical analysis
    noise1_flat = noise1_normalized.flatten()
    noise2_flat = noise2_normalized.flatten()
    
    # Create analysis charts
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Display first noise image
    axes[0, 0].imshow(noise1_img)
    axes[0, 0].set_title(f'Noise Image 1: {os.path.basename(noise1_path)}')
    axes[0, 0].axis('off')
    
    # 2. Display second noise image
    axes[0, 1].imshow(noise2_img)
    axes[0, 1].set_title(f'Noise Image 2: {os.path.basename(noise2_path)}')
    axes[0, 1].axis('off')
    
    # 3. Histogram comparison
    axes[1, 0].hist(noise1_flat, bins=50, alpha=0.7, label='Noise 1', density=True)
    axes[1, 0].hist(noise2_flat, bins=50, alpha=0.7, label='Noise 2', density=True)
    axes[1, 0].set_title('Noise Distribution Comparison')
    axes[1, 0].legend()
    axes[1, 0].set_xlabel('Pixel Value')
    axes[1, 0].set_ylabel('Density')
    
    # 4. Difference image
    diff_img = np.abs(noise1_normalized - noise2_normalized)
    axes[1, 1].imshow(diff_img, cmap='hot')
    axes[1, 1].set_title('Noise Difference Image')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'two_noises_comparison.png'), dpi=300)
    plt.close()
    
    # Calculate statistics
    noise1_stats = {
        'mean': np.mean(noise1_flat),
        'std': np.std(noise1_flat),
        'skew': stats.skew(noise1_flat),
        'kurtosis': stats.kurtosis(noise1_flat)
    }
    
    noise2_stats = {
        'mean': np.mean(noise2_flat),
        'std': np.std(noise2_flat),
        'skew': stats.skew(noise2_flat),
        'kurtosis': stats.kurtosis(noise2_flat)
    }
    
    # Kolmogorov-Smirnov test
    ks_statistic, ks_p_value = stats.ks_2samp(noise1_flat, noise2_flat)
    
    # Calculate mean squared error
    mse = np.mean((noise1_normalized - noise2_normalized) ** 2)
    
    # Print results
    print("Two Noise Images Comparison Results:")
    print("===================================")
    print(f"Noise 1 Statistics:")
    print(f"  Mean={noise1_stats['mean']:.4f}")
    print(f"  Std={noise1_stats['std']:.4f}")
    print(f"  Skewness={noise1_stats['skew']:.4f}")
    print(f"  Kurtosis={noise1_stats['kurtosis']:.4f}")
    print(f"\nNoise 2 Statistics:")
    print(f"  Mean={noise2_stats['mean']:.4f}")
    print(f"  Std={noise2_stats['std']:.4f}")
    print(f"  Skewness={noise2_stats['skew']:.4f}")
    print(f"  Kurtosis={noise2_stats['kurtosis']:.4f}")
    print(f"\nComparison Results:")
    print(f"  Mean Squared Error (MSE)={mse:.6f}")
    print(f"  Kolmogorov-Smirnov Test: Statistic={ks_statistic:.4f}, p-value={ks_p_value:.4f}")
    
    if ks_p_value > 0.05:
        print("\nConclusion: Cannot reject the hypothesis that the two distributions are the same (p > 0.05)")
        print("The two noises are not significantly different statistically")
    else:
        print("\nConclusion: The two distributions are significantly different (p <= 0.05)")
        print("The two noises are significantly different statistically")
    
    return {
        'noise1_stats': noise1_stats,
        'noise2_stats': noise2_stats,
        'mse': mse,
        'ks_statistic': ks_statistic,
        'ks_p_value': ks_p_value
    }

if __name__ == "__main__":
    # Example usage 1: Analyze similarity between a single noise image and Gaussian noise
    noise_path = "result/test3/x0_noise.npy"
    analyze_noise_vs_gaussian(noise_path)
    
    # Example usage 2: Compare two noise images
    # noise1_path = "path/to/your/first_noise.png"
    # noise2_path = "path/to/your/second_noise.png"
    # compare_two_noises(noise1_path, noise2_path)