import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def peaks_function(x, y):
    """Peaks函数"""
    term1 = 3 * (1 - x)**2 * np.exp(-(x**2 + (y + 1)**2))
    term2 = -10 * (x/5 - x**3 - y**5) * np.exp(-(x**2 + y**2))
    term3 = -(1/3) * np.exp(-((x + 1)**2 + y**2))
    return term1 + term2 + term3

class BPNetwork:
    
    def __init__(self, layer_sizes=[2, 64, 32, 1], seed=42):
        """
        初始化网络
        
        参数:
            layer_sizes: 各层节点数列表 [输入层, 隐藏层1, 隐藏层2, ..., 输出层]
            seed: 随机种子
        """
        np.random.seed(seed)
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes)
        
        # 使用He初始化权重
        self.weights = []
        self.biases = []
        for i in range(self.num_layers - 1):
            w = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(2.0 / layer_sizes[i])
            b = np.zeros((1, layer_sizes[i+1]))
            self.weights.append(w)
            self.biases.append(b)

    def relu(self, z):
        """ReLU激活函数"""
        return np.maximum(0, z)

    def relu_derivative(self, z):
        """ReLU导数"""
        return (z > 0).astype(float)
    
    def linear(self, z):
        """线性激活函数（用于回归输出层）"""
        return z
    
    def linear_derivative(self, z):
        """线性激活函数的导数"""
        return np.ones_like(z)
    
    def forward(self, X):
        """前向传播"""
        activations = [X]
        zs = []
        
        # 隐藏层使用ReLU
        for i in range(self.num_layers - 2):
            z = activations[-1].dot(self.weights[i]) + self.biases[i]
            zs.append(z)
            a = self.relu(z)
            activations.append(a)
        
        # 输出层使用线性激活（回归任务）
        z = activations[-1].dot(self.weights[-1]) + self.biases[-1]
        zs.append(z)
        a = self.linear(z)  # 回归任务使用线性输出
        activations.append(a)
        
        return activations, zs
    
    def backward(self, X, y, activations, zs, learning_rate):
        """反向传播"""
        m = X.shape[0]
        
        # 输出层误差（MSE损失的梯度）
        delta = (activations[-1] - y) * self.linear_derivative(zs[-1])
        
        # 反向传播
        deltas = [delta]
        for i in range(self.num_layers - 2, 0, -1):
            delta = delta.dot(self.weights[i].T) * self.relu_derivative(zs[i-1])
            deltas.insert(0, delta)
        
        # 更新权重和偏置
        for i in range(self.num_layers - 1):
            self.weights[i] -= learning_rate * activations[i].T.dot(deltas[i]) / m
            self.biases[i] -= learning_rate * np.sum(deltas[i], axis=0, keepdims=True) / m
    
    def train(self, X_train, y_train, epochs=10000, batch_size=32, 
              learning_rate=0.01, verbose=True, print_every=1000):
        """训练网络
        
        参数:
            X_train: 训练数据 (n_samples, n_features)
            y_train: 训练标签 (n_samples, 1)
            epochs: 训练轮数
            batch_size: 批量大小
            learning_rate: 学习率
            verbose: 是否打印训练信息
            print_every: 每隔多少轮打印一次
        """
        n_samples = X_train.shape[0]
        
        for epoch in range(1, epochs + 1):
            # 打乱训练数据
            indices = np.random.permutation(n_samples)
            X_train_shuffled = X_train[indices]
            y_train_shuffled = y_train[indices]
            
            # 小批量训练
            for i in range(0, n_samples, batch_size):
                X_batch = X_train_shuffled[i:i+batch_size]
                y_batch = y_train_shuffled[i:i+batch_size]
                
                # 前向传播和反向传播
                activations, zs = self.forward(X_batch)
                self.backward(X_batch, y_batch, activations, zs, learning_rate)
            
            # 打印训练信息
            if verbose and epoch % print_every == 0:
                activations, _ = self.forward(X_train)
                predictions = activations[-1]
                mse = np.mean((y_train - predictions) ** 2)
                print(f"Epoch {epoch}/{epochs}, MSE: {mse:.6f}")
    
    def predict(self, X):
        """预测"""
        activations, _ = self.forward(X)
        return activations[-1]


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


def bp_approximation():
    
    # 采样数据

    X_sample, z_sample = sample_peaks_data(n_samples=200, seed=42)
    
    # 数据归一化
    X_norm, z_norm, x_min, x_max, z_min, z_max = normalize_data(X_sample, z_sample)
    
    # 构建和训练BP网络
    net = BPNetwork(layer_sizes=[2, 64, 32, 1], seed=42)
    
    print(" 开始训练")
    net.train(X_norm, z_norm, epochs=10000, batch_size=32, 
              learning_rate=0.01, verbose=True, print_every=1000)
    
    # 预测
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
    
    # 可视化结果
    print("\n6. 绘制结果...")
    plot_comparison(X_sample, z_sample, X_grid, Y_grid, Z_true, Z_pred)
    
    # 计算误差
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
    
    max_error = np.max(np.abs(Z_true - Z_pred))
    print(f"\n 预测误差:")
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
    

def plot_comparison(X_sample, z_sample, X_grid, Y_grid, Z_true, Z_pred):
    """绘制采样点、真实函数和拟合结果的对比图"""
    
    fig = plt.figure(figsize=(18, 5))
    
    # 1. 采样点的3D散点图
    ax1 = fig.add_subplot(131, projection='3d')
    scatter = ax1.scatter(X_sample[:, 0], X_sample[:, 1], z_sample.ravel(), 
                         c='red', s=30, alpha=0.8)
    ax1.set_xlabel('x', fontsize=11)
    ax1.set_ylabel('y', fontsize=11)
    ax1.set_zlabel('f(x, y)', fontsize=11)
    ax1.set_title('sample point(200 sanple)', fontsize=13, fontweight='bold')
    ax1.view_init(elev=25, azim=45)
    
    # 2. 真实Peaks函数
    ax2 = fig.add_subplot(132, projection='3d')
    surf1 = ax2.plot_surface(X_grid, Y_grid, Z_true, cmap='coolwarm', 
                             alpha=0.9, edgecolor='none', antialiased=True)
    ax2.set_xlabel('x', fontsize=11)
    ax2.set_ylabel('y', fontsize=11)
    ax2.set_zlabel('f(x, y)', fontsize=11)
    ax2.set_title('true Peaks', fontsize=13, fontweight='bold')
    ax2.view_init(elev=25, azim=45)
    plt.colorbar(surf1, ax=ax2, shrink=0.5)
    
    # 3. BP网络拟合结果
    ax3 = fig.add_subplot(133, projection='3d')
    surf2 = ax3.plot_surface(X_grid, Y_grid, Z_pred, cmap='coolwarm', 
                             alpha=0.9, edgecolor='none', antialiased=True)
    # 叠加采样点
    ax3.scatter(X_sample[:, 0], X_sample[:, 1], z_sample.ravel(), 
               c='green', s=10, alpha=0.5, label='训练样本')
    ax3.set_xlabel('x', fontsize=11)
    ax3.set_ylabel('y', fontsize=11)
    ax3.set_zlabel('f(x, y)', fontsize=11)
    ax3.set_title('BP result', fontsize=13, fontweight='bold')
    ax3.view_init(elev=25, azim=45)
    plt.colorbar(surf2, ax=ax3, shrink=0.5)
    
    plt.tight_layout()
    plt.savefig('peaks_bp_approximation.png', dpi=150, bbox_inches='tight')
    plt.show()
    
 
if __name__ == '__main__':
    # 运行BP网络逼近
    bp_approximation()
