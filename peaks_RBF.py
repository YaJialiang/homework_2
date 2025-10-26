import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def peaks_function(x, y):
    """Peaks函数"""
    term1 = 3 * (1 - x)**2 * np.exp(-(x**2 + (y + 1)**2))
    term2 = -10 * (x/5 - x**3 - y**5) * np.exp(-(x**2 + y**2))
    term3 = -(1/3) * np.exp(-((x + 1)**2 + y**2))
    return term1 + term2 + term3


def k_means_clustering(X, k, max_iters=100, seed=42):
    """K-means聚类算法
    
    参数:
        X: 数据点 (n_samples, n_features)
        k: 聚类中心数量
        max_iters: 最大迭代次数
        seed: 随机种子
    
    返回:
        centers: 聚类中心 (k, n_features)
        labels: 每个样本的簇标签
    """
    np.random.seed(seed)
    n_samples, n_features = X.shape
    
    # 1. 随机初始化k个聚类中心
    indices = np.random.choice(n_samples, k, replace=False)
    centers = X[indices].copy()
    
    for iteration in range(max_iters):
        # 2. 计算每个样本到各中心的距离，分配到最近的中心
        distances = np.zeros((n_samples, k))
        for i in range(k):
            distances[:, i] = np.sum((X - centers[i])**2, axis=1)
        
        # 3. 为每个样本分配最近的聚类中心
        labels = np.argmin(distances, axis=1)
        
        # 4. 更新聚类中心为各簇的均值
        new_centers = np.zeros_like(centers)
        for i in range(k):
            cluster_points = X[labels == i]
            if len(cluster_points) > 0:
                new_centers[i] = cluster_points.mean(axis=0)
            else:
                new_centers[i] = centers[i]  # 保持原中心
        
        # 5. 检查收敛
        if np.allclose(centers, new_centers):
            break
        
        centers = new_centers
    
    # 重新计算最终的标签
    distances = np.zeros((n_samples, k))
    for i in range(k):
        distances[:, i] = np.sum((X - centers[i])**2, axis=1)
    labels = np.argmin(distances, axis=1)
    
    return centers, labels


class RBFNetwork:
    """RBF神经网络（用于函数逼近）
    
    网络结构: 输入层 -> RBF隐藏层 -> 线性输出层
    """
    
    def __init__(self, n_centers=20, use_kmeans=True, seed=42):
        """
        初始化RBF网络
        
        参数:
            n_centers: RBF中心数量
            use_kmeans: 是否使用K-means确定中心（True=广义RBF, False=正规化RBF）
            seed: 随机种子
        """
        np.random.seed(seed)
        self.n_centers = n_centers
        self.use_kmeans = use_kmeans
        self.centers = None
        self.sigmas = None  # 每个中心独立的方差
        self.weights = None
        
    def _gaussian_rbf(self, X, center_idx):
        """高斯径向基函数"""
        return np.exp(-np.sum((X - self.centers[center_idx])**2, axis=1) / (2 * self.sigmas[center_idx]**2))
    
    def fit(self, X_train, y_train):
        """训练RBF网络
        
        参数:
            X_train: 训练数据 (n_samples, n_features)
            y_train: 训练标签 (n_samples, 1)
        """
        n_samples = X_train.shape[0]
        
        # 1. 确定RBF中心
        if self.use_kmeans:
            # 广义RBF: 使用K-means聚类确定中心
            print(f"   使用K-means聚类确定{self.n_centers}个RBF中心...")
            self.centers, labels = k_means_clustering(X_train, self.n_centers)
            
            # 2. 为每个中心计算独立的方差参数
            self.sigmas = np.zeros(self.n_centers)
            for i in range(self.n_centers):
                cluster_points = X_train[labels == i]
                if len(cluster_points) > 0:
                    # 计算簇内样本到中心的平均距离
                    dists = np.sqrt(np.sum((cluster_points - self.centers[i])**2, axis=1))
                    self.sigmas[i] = np.mean(dists) if np.mean(dists) > 0 else 1.0
                else:
                    self.sigmas[i] = 1.0
            print(f"   σ 范围: [{self.sigmas.min():.4f}, {self.sigmas.max():.4f}]")
        else:
            # 正规化RBF: 使用训练样本作为中心
            print(f"   使用全部训练样本作为RBF中心（{n_samples}个）...")
            self.centers = X_train.copy()
            self.n_centers = n_samples
            
            # 计算所有中心之间的距离
            distances = []
            for i in range(self.n_centers):
                for j in range(i+1, self.n_centers):
                    dist = np.linalg.norm(self.centers[i] - self.centers[j])
                    distances.append(dist)
            
            # 使用统一的方差
            if distances:
                d_max = np.max(distances)
                sigma_uniform = 0.2
                # sigma_uniform = d_max / np.sqrt(3 * self.n_centers)
            else:
                sigma_uniform = 1.0
            
            self.sigmas = np.full(self.n_centers, sigma_uniform)
            print(f"   统一方差 σ = {sigma_uniform:.4f}")
        
        # 3. 计算隐藏层输出矩阵 Φ
        phi = np.zeros((n_samples, self.n_centers))
        for i in range(self.n_centers):
            phi[:, i] = self._gaussian_rbf(X_train, i)
        
        # 4. 使用伪逆求解输出层权重
        self.weights = np.linalg.pinv(phi).dot(y_train)
        
        # 计算训练误差
        y_pred = phi.dot(self.weights)
        mse = np.mean((y_train - y_pred)**2)
        print(f"   训练完成，训练MSE: {mse:.6f}")
    
    def predict(self, X):
        """预测
        
        参数:
            X: 测试数据 (n_samples, n_features)
        
        返回:
            预测值 (n_samples, 1)
        """
        n_samples = X.shape[0]
        
        # 计算隐藏层输出
        phi = np.zeros((n_samples, self.n_centers))
        for i in range(self.n_centers):
            phi[:, i] = self._gaussian_rbf(X, i)
        
        # 计算输出
        return phi.dot(self.weights)


def sample_peaks_data(n_samples=200, seed=42):
    """采样Peaks函数数据 """
    if seed is not None:
        np.random.seed(seed)
    
    # 计算每部分的样本数
    n_half = n_samples // 2
    
    # 第一半: 在 [-4, 4] × [-4, 4] 区域随机采样
    x1 = np.random.uniform(-3, 3, n_half)
    y1 = np.random.uniform(-3, 3, n_half)
    
    # 第二半: 在 [-2, 2] × [-1, 1] 区域随机采样（特征区域）
    x2 = np.random.uniform(-2, 2, n_samples - n_half)
    y2 = np.random.uniform(-1, 1, n_samples - n_half)
    
    # 合并两部分样本
    x = np.concatenate([x1, x2])
    y = np.concatenate([y1, y2])
    
    # 计算目标值
    z = peaks_function(x, y)
    
    # 组合输入
    X = np.column_stack([x, y])
    z = z.reshape(-1, 1)
    
    return X, z


def normalize_data(X, z):
    """归一化数据 """
    x_min, x_max = X.min(axis=0), X.max(axis=0)
    z_min, z_max = z.min(), z.max()
    
    # 归一化到[-1, 1]
    X_norm = 2 * (X - x_min) / (x_max - x_min + 1e-8) - 1
    z_norm = 2 * (z - z_min) / (z_max - z_min + 1e-8) - 1
    
    return X_norm, z_norm, x_min, x_max, z_min, z_max


def denormalize_output(z_norm, z_min, z_max):
    """反归一化输出"""
    return (z_norm + 1) / 2 * (z_max - z_min) + z_min


def rbf_approximation(use_kmeans=True, n_centers=30):
    """使用RBF网络逼近Peaks函数
    
    参数:
        use_kmeans: True=广义RBF, False=正规化RBF
        n_centers: RBF中心数量（仅对广义RBF有效）
    """
    network_type = "generalized RBF network" if use_kmeans else "normal RBF network"
    print(f"{network_type}逼近Peaks函数")
    
    # 1. 采样数据
    X_sample, z_sample = sample_peaks_data(n_samples=200, seed=42)
    
    # 2. 数据归一化
    X_norm, z_norm, x_min, x_max, z_min, z_max = normalize_data(X_sample, z_sample)
    
    # 3. 构建和训练RBF网络
    print(f"构建{network_type}...")
    if use_kmeans:
        print(f"   RBF中心数量: {n_centers} (K-means聚类)")
    else:
        print(f"   RBF中心数量: {len(X_sample)} (使用全部训练样本)")
    
    net = RBFNetwork(n_centers=n_centers, use_kmeans=use_kmeans, seed=42)
    
    net.fit(X_norm, z_norm)
    
    # 5. 预测
    x_test = np.linspace(-4, 4, 100)
    y_test = np.linspace(-4, 4, 100)
    X_grid, Y_grid = np.meshgrid(x_test, y_test)
    X_test_flat = np.column_stack([X_grid.ravel(), Y_grid.ravel()])
    
    # 归一化测试数据
    X_test_norm = 2 * (X_test_flat - x_min) / (x_max - x_min + 1e-8) - 1
    
    # 预测
    z_pred_norm = net.predict(X_test_norm)
    z_pred = denormalize_output(z_pred_norm, z_min, z_max)
    Z_pred = z_pred.reshape(X_grid.shape)
    
    # 真实值
    Z_true = peaks_function(X_grid, Y_grid)
    
    # 6. 可视化结果
    plot_comparison(X_sample, z_sample, X_grid, Y_grid, Z_true, Z_pred, network_type)
    
    # 7. 计算误差
    print("预测误差:")
    mse = np.mean((Z_true - Z_pred)**2)
    mae = np.mean(np.abs(Z_true - Z_pred))
    max_error = np.max(np.abs(Z_true - Z_pred))
    print(f"   MSE (均方误差): {mse:.6f}")
    print(f"   MAE (平均绝对误差): {mae:.6f}")
    print(f"   最大绝对误差: {max_error:.6f}")
    
    # 在采样点上的误差
    z_sample_pred_norm = net.predict(X_norm)
    z_sample_pred = denormalize_output(z_sample_pred_norm, z_min, z_max)
    sample_mse = np.mean((z_sample - z_sample_pred)**2)
    sample_mae = np.mean(np.abs(z_sample - z_sample_pred))
    print(f"\n   训练样本上的误差:")
    print(f"   MSE: {sample_mse:.6f}")
    print(f"   MAE: {sample_mae:.6f}")
    
    
    return mse, mae


def plot_comparison(X_sample, z_sample, X_grid, Y_grid, Z_true, Z_pred, network_type):
    """绘制采样点、真实函数和拟合结果的对比图"""
    
    fig = plt.figure(figsize=(18, 5))
    
    # 1. 采样点的3D散点图
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.scatter(X_sample[:, 0], X_sample[:, 1], z_sample.ravel(), 
               c='red', s=30, alpha=0.8)
    ax1.set_xlabel('x', fontsize=11)
    ax1.set_ylabel('y', fontsize=11)
    ax1.set_zlabel('f(x, y)', fontsize=11)
    ax1.set_title('Sample Points (200 samples)', fontsize=13, fontweight='bold')
    ax1.view_init(elev=25, azim=45)
    
    # 2. 真实Peaks函数
    ax2 = fig.add_subplot(132, projection='3d')
    surf1 = ax2.plot_surface(X_grid, Y_grid, Z_true, cmap='coolwarm', 
                             alpha=0.9, edgecolor='none', antialiased=True)
    ax2.set_xlabel('x', fontsize=11)
    ax2.set_ylabel('y', fontsize=11)
    ax2.set_zlabel('f(x, y)', fontsize=11)
    ax2.set_title('True Peaks Function', fontsize=13, fontweight='bold')
    ax2.view_init(elev=25, azim=45)
    plt.colorbar(surf1, ax=ax2, shrink=0.5)
    
    # 3. RBF网络拟合结果
    ax3 = fig.add_subplot(133, projection='3d')
    surf2 = ax3.plot_surface(X_grid, Y_grid, Z_pred, cmap='coolwarm', 
                             alpha=0.9, edgecolor='none', antialiased=True)
    # 叠加采样点
    ax3.scatter(X_sample[:, 0], X_sample[:, 1], z_sample.ravel(), 
               c='green', s=10, alpha=0.5)
    ax3.set_xlabel('x', fontsize=11)
    ax3.set_ylabel('y', fontsize=11)
    ax3.set_zlabel('f(x, y)', fontsize=11)
    ax3.set_title(f'{network_type} Result', fontsize=13, fontweight='bold')
    ax3.view_init(elev=25, azim=45)
    plt.colorbar(surf2, ax=ax3, shrink=0.5)
    
    plt.tight_layout()
    filename = f'peaks_rbf_{"generalized" if "广义" in network_type else "normalized"}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':

    # 广义RBF网络 (使用K-means)
    mse1, mae1 = rbf_approximation(use_kmeans=True, n_centers=30)
    # 正规化RBF网络 (使用全部样本作为中心)
    mse2, mae2 = rbf_approximation(use_kmeans=False)
    
    print(f"广义RBF (K-means, 30中心):  MSE={mse1:.6f}, MAE={mae1:.6f}")
    print(f"正规化RBF (全部样本中心):   MSE={mse2:.6f}, MAE={mae2:.6f}")


