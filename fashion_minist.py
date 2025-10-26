import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml


def load_fashion_mnist_data():
    """使用sklearn加载fashion_MNIST数据集"""
    print("正在加载fashion_MNIST数据集...")
    # 直接使用fetch_openml获取fashion_MNIST数据
    fashion_mnist = fetch_openml(name="Fashion-MNIST", version=1, as_frame=False)
    # 转换为numpy数组并归一化
    X = fashion_mnist.data.astype(np.float32) / 255.0
    y = fashion_mnist.target.astype(np.uint8)

    return X, y


def one_hot_encode(labels, num_classes=10):
    """将标签转换为one-hot编码"""
    return np.eye(num_classes)[labels]


class MNISTBPNetwork:
    """用于MNIST手写数字识别的BP神经网络
    
    网络结构：784 -> 128 -> 64 -> 10
    """
    
    def __init__(self, layer_sizes=[784, 128, 64, 10], seed=42):
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

    @staticmethod
    def softmax(z):
        """Softmax激活函数"""
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))  # 数值稳定性
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
    
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
        
        # 输出层使用Softmax
        z = activations[-1].dot(self.weights[-1]) + self.biases[-1]
        zs.append(z)
        a = self.softmax(z)
        activations.append(a)
        
        return activations, zs
    
    def backward(self, X, y, activations, zs, learning_rate):
        """反向传播"""
        m = X.shape[0]
        
        # 输出层误差（交叉熵损失的梯度）
        delta = activations[-1] - y  # (m, 10)
        
        # 反向传播
        deltas = [delta]
        for i in range(self.num_layers - 2, 0, -1):
            delta = delta.dot(self.weights[i].T) * self.relu_derivative(zs[i-1])
            deltas.insert(0, delta)
        
        # 更新权重和偏置
        for i in range(self.num_layers - 1):
            self.weights[i] -= learning_rate * activations[i].T.dot(deltas[i]) / m
            self.biases[i] -= learning_rate * np.sum(deltas[i], axis=0, keepdims=True) / m
    
    def train(self, X_train, y_train, X_val, y_val, epochs=20, batch_size=128, 
              learning_rate=0.01, verbose=True):
        """训练网络
        
        参数:
            X_train: 训练数据 (n_samples, 784)
            y_train: 训练标签 (n_samples, 10) one-hot编码
            X_val: 验证数据
            y_val: 验证标签
            epochs: 训练轮数
            batch_size: 批量大小
            learning_rate: 学习率
            verbose: 是否打印训练信息
        """
        n_samples = X_train.shape[0]
        
        for epoch in range(epochs):
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
            
            # 计算训练集和验证集的损失和准确率
            if verbose:
                train_loss, train_acc = self.evaluate(X_train, y_train)
                val_loss, val_acc = self.evaluate(X_val, y_val)
                
                print(f"Epoch {epoch+1}/{epochs} - "
                      f"train_loss: {train_loss:.4f}, train_acc: {train_acc:.4f}, "
                      f"val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f}")
        
    
    def predict(self, X):
        """预测"""
        activations, _ = self.forward(X)
        return np.argmax(activations[-1], axis=1)
    
    def evaluate(self, X, y):
        """评估模型（计算损失和准确率）"""
        activations, _ = self.forward(X)
        predictions = activations[-1]
        
        # 交叉熵损失
        m = X.shape[0]
        loss = -np.sum(y * np.log(predictions + 1e-8)) / m
        
        # 准确率
        pred_labels = np.argmax(predictions, axis=1)
        true_labels = np.argmax(y, axis=1)
        accuracy = np.mean(pred_labels == true_labels)
        
        return loss, accuracy



def main():
    X, y = load_fashion_mnist_data()
    
    # 数据已经是70000个样本，前60000是训练集，后10000是测试集
    X_train_full = X[:60000]
    y_train_full = y[:60000]
    X_test = X[60000:]
    y_test = y[60000:]
    
    
    # 使用50000个样本训练，10000个样本验证
    X_train = X_train_full[:50000]
    y_train = y_train_full[:50000]
    X_val = X_train_full[50000:]
    y_val = y_train_full[50000:]
    
    print(f"\n2. 数据划分：")
    print(f"   训练集: {X_train.shape[0]} 个样本")
    print(f"   验证集: {X_val.shape[0]} 个样本")
    print(f"   测试集: {X_test.shape[0]} 个样本")
    # One-hot编码
    y_train_onehot = one_hot_encode(y_train)
    y_val_onehot = one_hot_encode(y_val)
    y_test_onehot = one_hot_encode(y_test)
    
    layer_sizes = [784, 128, 64, 10]

    epochs = 40
    batch_size = 128
    learning_rate = 0.01

    print("\n开始训练...\n")
    model = MNISTBPNetwork(layer_sizes=layer_sizes, seed=42)
    model.train(X_train, y_train_onehot, X_val, y_val_onehot,
                epochs=epochs, batch_size=batch_size, 
                learning_rate=learning_rate, verbose=True)

    test_loss, test_acc = model.evaluate(X_test, y_test_onehot)
    print(f"   测试集损失: {test_loss:.4f}")
    print(f"   测试集准确率: {test_acc:.4f} ({test_acc*100:.2f}%)")
    
    # 预测
    predictions = model.predict(X_test)

    # print("\n各数字识别准确率：")
    # for digit in range(10):
    #     mask = (y_test == digit)
    #     digit_acc = np.mean(predictions[mask] == digit)
    #     print(f"   数字 {digit}: {digit_acc:.4f} ({digit_acc*100:.2f}%)")
    


if __name__ == '__main__':
    main()
