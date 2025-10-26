import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import PIL.Image as Image
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 用黑体显示中文
matplotlib.rcParams['axes.unicode_minus'] = False  # 正常显示负号


class AutoencoderBPNetwork:
    """用于图像压缩的BP自编码器网络
    
    网络结构：输入层 -> 隐藏层(压缩) -> 输出层(重构)
    例如：64 -> 16 -> 64 (将8x8=64维图像块压缩到16维再重构，压缩比4:1)
    """
    
    def __init__(self, layer_sizes=[64, 16, 64], seed=42):
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
    
def prepare_image_data(img, block_size=8):
    """
    将图像分割成块并归一化
    
    参数:
        img: PIL Image对象或numpy数组
        block_size: 图像块大小 (block_size x block_size)
    
    返回:
        blocks: 图像块数组 (n_blocks, block_size*block_size)
        img_array: 原始图像数组
        img_shape: 原始图像形状
    """
    # 转换为numpy数组
    if isinstance(img, Image.Image):
        img_array = np.array(img)
    else:
        img_array = img
    
    # 确保是灰度图
    if len(img_array.shape) == 3:
        img_array = np.mean(img_array, axis=2)
    
    height, width = img_array.shape
    
    # 计算需要的块数
    n_blocks_h = height // block_size
    n_blocks_w = width // block_size
    
    # 裁剪图像到可以被block_size整除的大小
    cropped_height = n_blocks_h * block_size
    cropped_width = n_blocks_w * block_size
    img_cropped = img_array[:cropped_height, :cropped_width]
    
    # 分割成块
    blocks = []
    for i in range(n_blocks_h):
        for j in range(n_blocks_w):
            block = img_cropped[i*block_size:(i+1)*block_size, 
                               j*block_size:(j+1)*block_size]
            blocks.append(block.flatten())
    
    blocks = np.array(blocks)
    
    # 归一化到[0,1]
    blocks = blocks / 255.0
    
    return blocks, img_array, (n_blocks_h, n_blocks_w, block_size)


def reconstruct_image(blocks, img_info):
    """
    从图像块重构图像
    
    参数:
        blocks: 图像块数组 (n_blocks, block_size*block_size)
        img_info: (n_blocks_h, n_blocks_w, block_size)
    
    返回:
        reconstructed_img: 重构的图像
    """
    n_blocks_h, n_blocks_w, block_size = img_info
    
    # 反归一化
    blocks = blocks * 255.0
    blocks = np.clip(blocks, 0, 255)
    
    # 重构图像
    img_reconstructed = np.zeros((n_blocks_h * block_size, n_blocks_w * block_size))
    
    block_idx = 0
    for i in range(n_blocks_h):
        for j in range(n_blocks_w):
            block = blocks[block_idx].reshape(block_size, block_size)
            img_reconstructed[i*block_size:(i+1)*block_size, 
                            j*block_size:(j+1)*block_size] = block
            block_idx += 1
    
    return img_reconstructed.astype(np.uint8)


def plot_image_compression(original_img, reconstructed_img, block_size, compression_ratio):
    """可视化图像压缩结果"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 原始图像
    axes[0].imshow(original_img, cmap='gray')
    axes[0].set_title('原始图像', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    # 重构图像
    axes[1].imshow(reconstructed_img, cmap='gray')
    axes[1].set_title(f'重构图像 (压缩比 {compression_ratio:.1f}:1)', fontsize=14, fontweight='bold')
    axes[1].axis('off')
    
    # 误差图
    error_img = np.abs(original_img[:reconstructed_img.shape[0], 
                                   :reconstructed_img.shape[1]].astype(float) - 
                      reconstructed_img.astype(float))
    im = axes[2].imshow(error_img, cmap='hot')
    axes[2].set_title(f'重构误差 (MAE={np.mean(error_img):.2f})', fontsize=14, fontweight='bold')
    axes[2].axis('off')
    plt.colorbar(im, ax=axes[2])
    
    plt.tight_layout()
    plt.savefig('image_compression.png', dpi=150, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    # 读取图像
    print("读取图像...")
    img = Image.open('lena.jpg')
    
    # 设置参数
    block_size = 8  # 8x8的图像块
    hidden_size = 16  # 压缩到16维
    input_size = block_size * block_size  # 64维输入
    compression_ratio = input_size / hidden_size  # 压缩比 4:1
    
    print(f"图像块大小: {block_size}x{block_size}")
    print(f"压缩维度: {input_size} -> {hidden_size} -> {input_size}")
    print(f"压缩比: {compression_ratio:.1f}:1")
    
    # 准备训练数据
    print("\n准备训练数据...")
    x_train, original_img, img_info = prepare_image_data(img, block_size)
    print(f"图像块数量: {x_train.shape[0]}")
    print(f"图像块维度: {x_train.shape[1]}")
    
    # 创建网络
    print(f"\n创建自编码器网络...")
    net = AutoencoderBPNetwork(layer_sizes=[input_size, hidden_size, input_size], seed=42)
    
    # 训练网络
    print(f"\n开始训练...")
    loss_history = net.train(x_train, epochs=2000, batch_size=10, 
                            learning_rate=0.05, verbose=True, print_every=200)
    
    # 重构图像
    print(f"\n重构图像...")
    x_reconstructed = net.predict(x_train)
    reconstructed_img = reconstruct_image(x_reconstructed, img_info)
    
    # 计算重构误差
    original_cropped = original_img[:reconstructed_img.shape[0], :reconstructed_img.shape[1]]
    mse = np.mean((original_cropped - reconstructed_img.astype(float))**2)
    mae = np.mean(np.abs(original_cropped - reconstructed_img.astype(float)))
    psnr = 10 * np.log10(255**2 / mse) if mse > 0 else float('inf')
    
    print(f"\n图像质量评估:") 
    
    print(f"  MSE: {mse:.4f}")
    print(f"  MAE: {mae:.4f}")
    print(f"  PSNR: {psnr:.2f} dB")
    
    # 可视化结果
    print(f"\n生成可视化图像...")
    plot_image_compression(original_img, reconstructed_img, block_size, compression_ratio)
    
    # 绘制训练损失曲线
    plot_training_loss(loss_history)
    
    # 显示压缩效果
    print(f"\n压缩效果:")
    print(f"  原始数据大小: {x_train.shape[0]} blocks × {input_size} = {x_train.shape[0] * input_size} 像素")
    print(f"  压缩后大小: {x_train.shape[0]} blocks × {hidden_size} = {x_train.shape[0] * hidden_size} 特征")
    print(f"  压缩比: {compression_ratio:.1f}:1")
    print(f"  空间节省: {(1 - 1/compression_ratio) * 100:.1f}%")
