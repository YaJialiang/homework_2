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

    def __init__(self, n_input=2, n_hidden=4, n_output=1, seed=None, output_activation='tanh'):
        if seed is not None:
            np.random.seed(seed)
        # w1: (n_input + 1) x n_hidden  -- maps [bias, x] -> hidden
        # w2: (n_hidden + 1) x n_output -- maps [bias, hidden] -> output
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.w1 = 2 * np.random.random((n_input + 1, n_hidden)) - 1
        self.w2 = 2 * np.random.random((n_hidden + 1, n_output)) - 1
        # output_activation: 'sigmoid' for targets in [0,1], 'tanh' for targets in [-1,1]
        if output_activation not in ('sigmoid', 'tanh'):
            raise ValueError("output_activation must be 'sigmoid' or 'tanh'")
        self.output_activation = output_activation

    @staticmethod
    def sigmoid(z):
        return 1.0 / (1.0 + np.exp(-z))

    @staticmethod
    def sigmoid_deriv(o):
        return o * (1.0 - o)

    @staticmethod
    def tanh(z):
        return np.tanh(z)

    @staticmethod
    def tanh_deriv(o):
        return 1.0 - o * o

    def forward(self, X):
        """Forward pass.
        X: shape (m, n_input) -- inputs without bias
        returns: (layer1, layer2, layer3)
          layer1: (m, n_input+1) inputs with bias as first column (value -1)
          layer2: (m, n_hidden+1) hidden activations with bias first col
          layer3: (m, n_output) outputs
        """
        m = X.shape[0]
        # add bias column (-1) at start
        layer1 = np.hstack([-np.ones((m, 1)), X])
        z2 = np.dot(layer1, self.w1)
        a2 = self.sigmoid(z2)
        layer2 = np.hstack([-np.ones((m, 1)), a2])
        z3 = np.dot(layer2, self.w2)
        if self.output_activation == 'sigmoid':
            a3 = self.sigmoid(z3)
        else:
            a3 = self.tanh(z3)
        return layer1, layer2, a3

    def train(self, X, T, epochs=10000, eta=0.3, verbose=True, print_every=1000):
        """T
    X: (m, n_input) inputs (no bias column)
    T: (m, n_output) targets in range [0,1] for sigmoid or [-1,1] for tanh
        """
        X = np.asarray(X, dtype=float)
        T = np.asarray(T, dtype=float)
        errors=[]
        for epoch in range(1, epochs + 1):
            layer1, layer2, out = self.forward(X)

            # output error term
            if self.output_activation == 'sigmoid':
                delta3 = (out - T) * self.sigmoid_deriv(out)  # (m, n_output)
            else:
                delta3 = (out - T) * self.tanh_deriv(out)
            # hidden error term (exclude bias weight when backpropagating)
            # w2[1:,:] has shape (n_hidden, n_output)
            delta2 = delta3.dot(self.w2[1:, :].T) * self.sigmoid_deriv(layer2[:, 1:])
            m=X.shape[0]
            # gradient descent updates (batch)
            self.w2 -= eta * (layer2.T.dot(delta3)) / m
            self.w1 -= eta * (layer1.T.dot(delta2)) /m
            if verbose and epoch % print_every == 0:
                mse = np.mean(0.5 * np.square(T - out))
                errors.append(mse)
                print(f"Epoch {epoch}, MSE: {mse:.6f}")
        return errors

    def predict(self, X):
        _, _, out = self.forward(np.asarray(X, dtype=float))
        return out


def plot_peaks_3d():
    """绘制Peaks函数的3D曲面图"""
    # 创建网格
    x = np.linspace(-4, 4, 100)
    y = np.linspace(-4, 4, 100)
    X, Y = np.meshgrid(x, y)
    Z = peaks_function(X, Y)
    
    # 创建3D图
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8, 
                           edgecolor='none', antialiased=True)
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.set_zlabel('f(x, y)', fontsize=12)
    ax.set_title('Peaks Function - 3D Surface', fontsize=14, fontweight='bold')
    ax.view_init(elev=30, azim=45)
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    
    plt.tight_layout()
    plt.show()


def sample_peaks_data(n_samples=200, x_range=(-4, 4), y_range=(-4, 4), seed=42):
    """在指定区域随机采样Peaks函数数据
    
    参数:
        n_samples: 采样点数量
        x_range: x的采样范围
        y_range: y的采样范围
        seed: 随机种子
    
    返回:
        X: (n_samples, 2) 输入数据 [x, y]
        z: (n_samples, 1) 目标值 f(x,y)
    """
    if seed is not None:
        np.random.seed(seed)
    
    # 在[-4, 4] × [-4, 4]区域随机采样
    x = np.random.uniform(x_range[0], x_range[1], n_samples)
    y = np.random.uniform(y_range[0], y_range[1], n_samples)
    
    # 计算目标值
    z = peaks_function(x, y)
    
    # 组合输入
    X = np.column_stack([x, y])
    z = z.reshape(-1, 1)
    
    return X, z


def normalize_data(X, z):
    """归一化数据到[-1, 1]范围
    
    参数:
        X: 输入数据
        z: 目标值
    
    返回:
        X_norm: 归一化后的输入
        z_norm: 归一化后的目标值
        x_min, x_max: 输入的最小最大值
        z_min, z_max: 目标值的最小最大值
    """
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
    """使用BP网络逼近Peaks函数"""
    print("=" * 80)
    print("BP神经网络逼近Peaks函数")
    print("=" * 80)
    
    # 1. 采样数据
    print("\n1. 在区域 [-4, 4] × [-4, 4] 内随机采样 200 个点...")
    X_sample, z_sample = sample_peaks_data(n_samples=200, seed=42)
    print(f"   采样完成: X shape = {X_sample.shape}, z shape = {z_sample.shape}")
    print(f"   z 范围: [{z_sample.min():.4f}, {z_sample.max():.4f}]")
    
    # 2. 数据归一化
    print("\n2. 数据归一化到 [-1, 1]...")
    X_norm, z_norm, x_min, x_max, z_min, z_max = normalize_data(X_sample, z_sample)
    print(f"   归一化后 z 范围: [{z_norm.min():.4f}, {z_norm.max():.4f}]")
    
    # 3. 构建和训练BP网络
    print("\n3. 构建BP神经网络...")
    print("   网络结构: 2 输入 -> 30 隐藏层 -> 1 输出")
    print("   激活函数: 隐藏层=Sigmoid, 输出层=Tanh")
    
    net = BPNetwork(n_input=2, n_hidden=30, n_output=1, seed=42, output_activation='tanh')
    
    print("\n4. 开始训练...")
    errors = net.train(X_norm, z_norm, epochs=50000, eta=0.5, verbose=True, print_every=5000)
    
    # 4. 预测
    print("\n5. 在整个区域生成预测...")
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
    print("\n6. 绘制结果...")
    plot_comparison(X_sample, z_sample, X_grid, Y_grid, Z_true, Z_pred)
    
    # 7. 计算误差
    mse = np.mean((Z_true - Z_pred)**2)
    mae = np.mean(np.abs(Z_true - Z_pred))
    max_error = np.max(np.abs(Z_true - Z_pred))
    print(f"\n7. 预测误差:")
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
    
    print("\n" + "=" * 80)
    print("完成！")
    print("=" * 80)


def plot_comparison(X_sample, z_sample, X_grid, Y_grid, Z_true, Z_pred):
    """绘制采样点、真实函数和拟合结果的对比图"""
    
    fig = plt.figure(figsize=(18, 5))
    
    # 1. 采样点的3D散点图
    ax1 = fig.add_subplot(131, projection='3d')
    scatter = ax1.scatter(X_sample[:, 0], X_sample[:, 1], z_sample, 
                         c=z_sample, cmap='viridis', s=30, alpha=0.8)
    ax1.set_xlabel('x', fontsize=11)
    ax1.set_ylabel('y', fontsize=11)
    ax1.set_zlabel('f(x, y)', fontsize=11)
    ax1.set_title('采样点 (200个样本)', fontsize=13, fontweight='bold')
    ax1.view_init(elev=25, azim=45)
    plt.colorbar(scatter, ax=ax1, shrink=0.5)
    
    # 2. 真实Peaks函数
    ax2 = fig.add_subplot(132, projection='3d')
    surf1 = ax2.plot_surface(X_grid, Y_grid, Z_true, cmap='viridis', 
                             alpha=0.9, edgecolor='none', antialiased=True)
    ax2.set_xlabel('x', fontsize=11)
    ax2.set_ylabel('y', fontsize=11)
    ax2.set_zlabel('f(x, y)', fontsize=11)
    ax2.set_title('真实Peaks函数', fontsize=13, fontweight='bold')
    ax2.view_init(elev=25, azim=45)
    plt.colorbar(surf1, ax=ax2, shrink=0.5)
    
    # 3. BP网络拟合结果
    ax3 = fig.add_subplot(133, projection='3d')
    surf2 = ax3.plot_surface(X_grid, Y_grid, Z_pred, cmap='viridis', 
                             alpha=0.9, edgecolor='none', antialiased=True)
    # 叠加采样点
    ax3.scatter(X_sample[:, 0], X_sample[:, 1], z_sample, 
               c='red', s=10, alpha=0.5, label='训练样本')
    ax3.set_xlabel('x', fontsize=11)
    ax3.set_ylabel('y', fontsize=11)
    ax3.set_zlabel('f(x, y)', fontsize=11)
    ax3.set_title('BP网络拟合结果', fontsize=13, fontweight='bold')
    ax3.view_init(elev=25, azim=45)
    plt.colorbar(surf2, ax=ax3, shrink=0.5)
    
    plt.tight_layout()
    plt.savefig('peaks_bp_approximation.png', dpi=150, bbox_inches='tight')
    print("   ✓ 对比图已保存为 'peaks_bp_approximation.png'")
    plt.show()
    
    # 绘制误差分布图
    fig2, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 误差的2D热图
    error = np.abs(Z_true - Z_pred)
    im1 = axes[0].contourf(X_grid, Y_grid, error, levels=20, cmap='hot')
    axes[0].scatter(X_sample[:, 0], X_sample[:, 1], c='blue', s=10, alpha=0.5)
    axes[0].set_xlabel('x', fontsize=11)
    axes[0].set_ylabel('y', fontsize=11)
    axes[0].set_title('绝对误差分布（红色=高误差）', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    plt.colorbar(im1, ax=axes[0])
    
    # 误差的3D图
    ax_err = fig2.add_subplot(122, projection='3d')
    surf_err = ax_err.plot_surface(X_grid, Y_grid, error, cmap='hot', 
                                    alpha=0.9, edgecolor='none')
    ax_err.set_xlabel('x', fontsize=11)
    ax_err.set_ylabel('y', fontsize=11)
    ax_err.set_zlabel('|误差|', fontsize=11)
    ax_err.set_title('预测误差3D视图', fontsize=12, fontweight='bold')
    ax_err.view_init(elev=25, azim=45)
    plt.colorbar(surf_err, ax=ax_err, shrink=0.5)
    
    plt.tight_layout()
    plt.savefig('peaks_error_analysis.png', dpi=150, bbox_inches='tight')
    print("   ✓ 误差分析图已保存为 'peaks_error_analysis.png'")
    plt.show()


if __name__ == '__main__':
    # 运行BP网络逼近
    bp_approximation()
