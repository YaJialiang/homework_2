import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 用黑体显示中文
matplotlib.rcParams['axes.unicode_minus'] = False  # 正常显示负号


class AutoencoderBPNetwork:
    """用于字母压缩的BP自编码器网络
    
    网络结构：输入层 -> 隐藏层(压缩) -> 输出层(重构)
    例如：63 -> 6 -> 63 (将63维数据压缩到6维再重构)
    """
    
    def __init__(self, layer_sizes=[63, 6, 63], seed=42):
        """
        初始化自编码器网络
        
        参数:
            layer_sizes: 各层节点数列表 [输入层, 隐藏层(编码层), 输出层]
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

    def sigmoid(self, z):
        """Sigmoid激活函数（适合0-1数据）"""
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))  # 防止溢出

    def sigmoid_derivative(self, z):
        """Sigmoid导数"""
        s = self.sigmoid(z)
        return s * (1 - s)

    def relu(self, z):
        """ReLU激活函数"""
        return np.maximum(0, z)

    def relu_derivative(self, z):
        """ReLU导数"""
        return (z > 0).astype(float)
    
    def forward(self, X):
        """前向传播"""
        activations = [X]
        zs = []
        
        # 隐藏层使用Sigmoid
        for i in range(self.num_layers - 2):
            z = activations[-1].dot(self.weights[i]) + self.biases[i]
            zs.append(z)
            a = self.sigmoid(z)
            activations.append(a)
        
        # 输出层也使用Sigmoid（因为输出是0-1值）
        z = activations[-1].dot(self.weights[-1]) + self.biases[-1]
        zs.append(z)
        a = self.sigmoid(z)
        activations.append(a)
        
        return activations, zs
    
    def backward(self, X, y, activations, zs, learning_rate):
        """反向传播（使用MSE损失）"""
        m = X.shape[0]
        
        # 输出层误差（MSE损失的梯度）
        delta = (activations[-1] - y) * self.sigmoid_derivative(zs[-1])
        
        # 反向传播
        deltas = [delta]
        for i in range(self.num_layers - 2, 0, -1):
            delta = delta.dot(self.weights[i].T) * self.sigmoid_derivative(zs[i-1])
            deltas.insert(0, delta)
        
        # 更新权重和偏置
        for i in range(self.num_layers - 1):
            self.weights[i] -= learning_rate * activations[i].T.dot(deltas[i]) / m
            self.biases[i] -= learning_rate * np.sum(deltas[i], axis=0, keepdims=True) / m
    
    def train(self, X_train, epochs=5000, batch_size=1, learning_rate=0.1, 
              verbose=True, print_every=500):
        """训练自编码器网络
        
        参数:
            X_train: 训练数据 (n_samples, input_dim)
            epochs: 训练轮数
            batch_size: 批量大小
            learning_rate: 学习率
            verbose: 是否打印训练信息
            print_every: 每多少轮打印一次
        """
        n_samples = X_train.shape[0]
        loss_history = []
        
        for epoch in range(epochs):
            # 打乱训练数据
            indices = np.random.permutation(n_samples)
            X_train_shuffled = X_train[indices]
            
            # 小批量训练
            for i in range(0, n_samples, batch_size):
                X_batch = X_train_shuffled[i:i+batch_size]
                
                # 前向传播和反向传播（自编码器：输入=目标输出）
                activations, zs = self.forward(X_batch)
                self.backward(X_batch, X_batch, activations, zs, learning_rate)
            
            # 计算并记录损失
            activations, _ = self.forward(X_train)
            predictions = activations[-1]
            mse = np.mean((X_train - predictions) ** 2)
            loss_history.append(mse)
            
            # 打印训练信息
            if verbose and (epoch + 1) % print_every == 0:
                print(f"Epoch {epoch+1}/{epochs} - MSE: {mse:.6f}")
        
        return loss_history
    
    def predict(self, X):
        """预测（重构）"""
        activations, _ = self.forward(X)
        return activations[-1]
    
    def encode(self, X):
        """编码：获取压缩后的特征表示"""
        activation = X
        # 只进行到隐藏层（编码层）
        for i in range(len(self.weights) // 2):
            z = activation.dot(self.weights[i]) + self.biases[i]
            activation = self.sigmoid(z)
        return activation
    
    def decode(self, encoded):
        """解码：从压缩特征重构数据"""
        activation = encoded
        # 从隐藏层到输出层
        for i in range(len(self.weights) // 2, len(self.weights)):
            z = activation.dot(self.weights[i]) + self.biases[i]
            activation = self.sigmoid(z)
        return activation


def plot_letters(original, reconstructed, letters_names, save_name='letter_compression.png'):
    """可视化字母压缩前后的对比
    
    参数:
        original: 原始字母数据 (n_letters, 128)
        reconstructed: 重构后的字母数据 (n_letters, 128)
        letters_names: 字母名称列表
        save_name: 保存的文件名
    """
    n_letters = len(original)
    
    # 将128维向量reshape为8x16的图像（8行16列）
    fig, axes = plt.subplots(3, n_letters, figsize=(2*n_letters, 6))
    
    for i in range(n_letters):
        # 原始字母
        original_img = np.array(original[i]).reshape(8, 16)
        axes[0, i].imshow(original_img, cmap='binary', interpolation='nearest')
        axes[0, i].set_title(f'{letters_names[i]}', fontsize=14, fontweight='bold')
        axes[0, i].axis('off')
        
        # 重构字母
        reconstructed_img = reconstructed[i].reshape(8, 16)
        axes[1, i].imshow(reconstructed_img, cmap='binary', interpolation='nearest')
        axes[1, i].set_title('重构', fontsize=12)
        axes[1, i].axis('off')
        
        # 误差图
        error_img = np.abs(original_img - reconstructed_img)
        im = axes[2, i].imshow(error_img, cmap='hot', interpolation='nearest', vmin=0, vmax=1)
        axes[2, i].set_title(f'误差: {np.mean(error_img):.3f}', fontsize=10)
        axes[2, i].axis('off')
    
    # 添加行标签
    axes[0, 0].text(-0.5, 3.5, '原始', fontsize=14, fontweight='bold', 
                    ha='right', va='center', rotation=90, transform=axes[0, 0].transData)
    axes[1, 0].text(-0.5, 3.5, '重构', fontsize=14, fontweight='bold', 
                    ha='right', va='center', rotation=90, transform=axes[1, 0].transData)
    axes[2, 0].text(-0.5, 3.5, '误差', fontsize=14, fontweight='bold', 
                    ha='right', va='center', rotation=90, transform=axes[2, 0].transData)
    
    plt.tight_layout()
    plt.savefig(save_name, dpi=150, bbox_inches='tight')
    plt.show()


def plot_training_loss(loss_history):
    """绘制训练损失曲线"""
    plt.figure(figsize=(10, 5))
    plt.plot(loss_history, linewidth=2)
    plt.xlabel('Epoch', fontsize=12, fontweight='bold')
    plt.ylabel('MSE Loss', fontsize=12, fontweight='bold')
    plt.title('训练损失曲线', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.savefig('training_loss.png', dpi=150, bbox_inches='tight')
    plt.show()
    
if __name__ == '__main__':
    
    # 定义字母数据 (8x16的点阵，共128个像素)
    # 每个字母是8行16列的点阵图
    A = [
        0,0,0,0,1,1,1,1,1,1,0,0,0,0,0,0,
        0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,
        0,0,1,1,1,0,0,0,0,1,1,1,0,0,0,0,
        0,1,1,1,0,0,0,0,0,0,1,1,1,0,0,0,
        0,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,
        1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,
        1,1,1,0,0,0,0,0,0,0,0,1,1,1,0,0,
        1,1,0,0,0,0,0,0,0,0,0,0,1,1,0,0
    ]
    
    B = [
        1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,
        1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,
        1,1,1,0,0,0,0,0,0,1,1,1,0,0,0,0,
        1,1,1,0,0,0,0,0,0,1,1,1,0,0,0,0,
        1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,
        1,1,1,0,0,0,0,0,0,1,1,1,1,0,0,0,
        1,1,1,0,0,0,0,0,0,0,1,1,1,0,0,0,
        1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0
    ]
    
    C = [
        0,0,0,1,1,1,1,1,1,1,1,1,0,0,0,0,
        0,0,1,1,1,1,1,1,1,1,1,1,1,0,0,0,
        0,1,1,1,0,0,0,0,0,0,0,1,1,1,0,0,
        1,1,1,0,0,0,0,0,0,0,0,0,1,1,0,0,
        1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,
        1,1,1,0,0,0,0,0,0,0,0,0,1,1,0,0,
        0,1,1,1,0,0,0,0,0,0,0,1,1,1,0,0,
        0,0,1,1,1,1,1,1,1,1,1,1,1,0,0,0
    ]
    
    D = [
        1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,
        1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,
        1,1,1,0,0,0,0,0,0,0,1,1,1,1,0,0,
        1,1,1,0,0,0,0,0,0,0,0,1,1,1,0,0,
        1,1,1,0,0,0,0,0,0,0,0,1,1,1,0,0,
        1,1,1,0,0,0,0,0,0,0,1,1,1,1,0,0,
        1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,
        1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0
    ]
    
    E = [
        1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,
        1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,
        1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,
        1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,
        1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,
        1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,
        1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,
        1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0
    ]
    
    F = [
        1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,
        1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,
        1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,
        1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,
        1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,
        1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,
        1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,
        1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0
    ]
    
    G = [
        0,0,0,1,1,1,1,1,1,1,1,1,0,0,0,0,
        0,0,1,1,1,1,1,1,1,1,1,1,1,0,0,0,
        0,1,1,1,0,0,0,0,0,0,0,1,1,1,0,0,
        1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,
        1,1,1,0,0,0,0,1,1,1,1,1,1,1,0,0,
        1,1,1,0,0,0,0,0,0,0,0,1,1,1,0,0,
        0,1,1,1,0,0,0,0,0,0,0,1,1,1,0,0,
        0,0,1,1,1,1,1,1,1,1,1,1,1,0,0,0
    ]
    
    H = [
        1,1,1,0,0,0,0,0,0,0,0,1,1,1,0,0,
        1,1,1,0,0,0,0,0,0,0,0,1,1,1,0,0,
        1,1,1,0,0,0,0,0,0,0,0,1,1,1,0,0,
        1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,
        1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,
        1,1,1,0,0,0,0,0,0,0,0,1,1,1,0,0,
        1,1,1,0,0,0,0,0,0,0,0,1,1,1,0,0,
        1,1,1,0,0,0,0,0,0,0,0,1,1,1,0,0
    ]
    
    I = [
        0,0,1,1,1,1,1,1,1,1,1,1,1,0,0,0,
        0,0,1,1,1,1,1,1,1,1,1,1,1,0,0,0,
        0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,
        0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,
        0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,
        0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,
        0,0,1,1,1,1,1,1,1,1,1,1,1,0,0,0,
        0,0,1,1,1,1,1,1,1,1,1,1,1,0,0,0
    ]
    
    J = [
        0,0,0,0,0,0,0,1,1,1,1,1,1,1,0,0,
        0,0,0,0,0,0,0,1,1,1,1,1,1,1,0,0,
        0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,
        0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,
        0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,
        1,1,1,0,0,0,0,0,0,0,1,1,1,0,0,0,
        1,1,1,1,0,0,0,0,0,1,1,1,0,0,0,0,
        0,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0
    ]

    # 准备训练数据
    x_train = np.array([A, B, C, D, E, F, G, H, I, J], dtype=np.float32)
    letters_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
    
    
    # 构建自编码器网络: 128维 -> 16维 -> 128维

    net = AutoencoderBPNetwork(layer_sizes=[128, 4, 128], seed=42)
    
    # 训练网络
    print(f"开始训练...")

    loss_history = net.train(x_train, epochs=5000, batch_size=1, 
                            learning_rate=0.1, verbose=True, print_every=500)

 
    # 重构所有字母
    x_reconstructed = net.predict(x_train)
    
    # 获取压缩编码
    x_encoded = net.encode(x_train)
    
    # 计算重构误差
    print(f"\n重构误差分析:")
    for i, name in enumerate(letters_names):
        mse = np.mean((x_train[i] - x_reconstructed[i])**2)
        mae = np.mean(np.abs(x_train[i] - x_reconstructed[i]))
        print(f"  {name}: MSE={mse:.6f}, MAE={mae:.6f}")
    
    overall_mse = np.mean((x_train - x_reconstructed)**2)
    overall_mae = np.mean(np.abs(x_train - x_reconstructed))
    print(f"\n  总体: MSE={overall_mse:.6f}, MAE={overall_mae:.6f}")
    
    # 可视化结果
    print(f"\n生成可视化图像...")
    plot_letters(x_train, x_reconstructed, letters_names)
    
    # 绘制训练损失曲线
    plot_training_loss(loss_history)
    
    print(f"重构误差MSE: {overall_mse:.6f}")
