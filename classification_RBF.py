import numpy as np
from sklearn.cluster import KMeans


class NormalizedRBFNetwork:
    """正规化RBF网络 (Normalized RBF Network)

    正规化网络的隐单元就是训练样本，所以正规化网络的隐单元个数与训练样本的个数相同。
    每个训练样本对应一个RBF中心，中心就是训练样本本身。
    """

    def __init__(self, n_input=2, n_output=3, seed=None):
        self.n_input = n_input
        self.n_output = n_output
        self.n_hidden = None  # 将在训练时设置为样本数
        if seed is not None:
            np.random.seed(seed)
        self.centers = None
        self.sigma = 1.0
        self.w = None

    def _rbf(self, X):
        """Compute RBF (Gaussian) activations for all samples"""
        G = np.zeros((X.shape[0], self.n_hidden))
        for i in range(self.n_hidden):
            G[:, i] = np.exp(-np.sum((X - self.centers[i])**2, axis=1) / (2 * self.sigma**2))
        return G

    def fit(self, X, T):
        """Train normalized RBF network
        
        正规化网络：隐单元个数 = 训练样本个数
        中心直接使用训练样本作为RBF中心
        """
        # 1. 隐单元个数等于训练样本个数
        self.n_hidden = X.shape[0]
        
        # 2. 中心就是训练样本本身
        self.centers = X.copy()

        # 3. 计算宽度参数sigma (使用样本间的平均距离)
        dists = []
        for i in range(len(X)):
            for j in range(i+1, len(X)):
                dists.append(np.linalg.norm(X[i] - X[j]))
        if len(dists) > 0:
            self.sigma = np.mean(dists) / np.sqrt(2 * self.n_hidden)
        else:
            self.sigma = 1.0

        # 4. 计算基函数激活值
        G = self._rbf(X)

        # 5. 使用伪逆求解输出权重
        self.w = np.linalg.pinv(G).dot(T)

    def predict(self, X):
        G = self._rbf(X)
        return G.dot(self.w)


class GeneralizedRBFNetwork:
    """广义RBF网络 (Generalized RBF Network)
    
    广义网络：
    - 输入层有 M 个神经元
    - 隐层有 I (I<M, 即隐层神经元少于输入维度) 个神经元，每个有自己的中心和方差
    - 输出层有 J 个神经元
    - 隐层到输出层有权系数矩阵（需要训练）
    
    使用聚类方法（如K-means）确定隐层中心
    """

    def __init__(self, n_input=2, n_hidden=4, n_output=3, seed=None):
        """
        n_input: M, 输入维度
        n_hidden: I, 隐层神经元个数（通常 I < n_input 对于广义RBF，但对分类问题可以 I < 样本数）
        n_output: J, 输出神经元个数
        """
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output
        if seed is not None:
            np.random.seed(seed)
        self.centers = None
        # 每个隐层神经元有自己的方差参数
        self.sigmas = None
        # 隐层到输出层的权系数矩阵
        self.w = None

    def _rbf(self, X):
        """Compute RBF activations with individual sigmas for each center"""
        G = np.zeros((X.shape[0], self.n_hidden))
        for i in range(self.n_hidden):
            G[:, i] = np.exp(-np.sum((X - self.centers[i])**2, axis=1) / (2 * self.sigmas[i]**2))
        return G

    def fit(self, X, T):
        """Train generalized RBF network
        
        使用聚类确定隐层中心，每个中心有独立的方差参数
        """
        # 1. 使用KMeans聚类确定隐层中心（I个中心，I < 样本数）
        kmeans = KMeans(n_clusters=self.n_hidden, random_state=0).fit(X)
        self.centers = kmeans.cluster_centers_
        
        # 2. 为每个隐层神经元计算独立的方差参数
        self.sigmas = np.zeros(self.n_hidden)
        labels = kmeans.labels_
        
        for i in range(self.n_hidden):
            # 找到属于第i个簇的所有样本
            cluster_points = X[labels == i]
            if len(cluster_points) > 0:
                # 计算簇内样本到中心的平均距离作为该隐层神经元的方差
                dists = np.sqrt(np.sum((cluster_points - self.centers[i])**2, axis=1))
                self.sigmas[i] = np.mean(dists) if np.mean(dists) > 0 else 1.0
            else:
                self.sigmas[i] = 1.0

        # 3. 计算隐层激活值（RBF函数）
        G = self._rbf(X)

        # 4. 使用伪逆求解隐层到输出层的权系数矩阵
        self.w = np.linalg.pinv(G).dot(T)

    def predict(self, X):
        G = self._rbf(X)
        return G.dot(self.w)


# --- 测试三种RBF网络 ---
if __name__ == '__main__':
    X = np.array([[0.75, 1.0], [0.5, 0.75], [0.25, 0.0], [0.5, 0.0],
                  [0.0, 0.0], [1.0, 0.75], [1.0, 1.0], [0.5, 0.25], [0.75, 0.5]])
    T = np.array([[1, -1, -1], [1, -1, -1], [1, -1, -1],
                  [-1, 1, -1], [-1, 1, -1], [-1, 1, -1],
                  [-1, -1, 1], [-1, -1, 1], [-1, -1, 1]])


    net2 = NormalizedRBFNetwork(n_input=2, n_output=3, seed=1)
    net2.fit(X, T)
    preds2 = net2.predict(X)
    
    print(f"隐层神经元数(=样本数): {net2.n_hidden}")
    print(f"Centers shape: {net2.centers.shape}")
    print(f"统一的Sigma: {net2.sigma:.4f}")
    print("\nPredictions:")
    correct2 = 0
    for i, (x, t, p) in enumerate(zip(X, T, preds2)):
        pred_class = np.argmax(p)
        true_class = np.argmax(t)
        correct2 += (pred_class == true_class)
        p_formatted = [f"{val:7.4f}" for val in p]
        print(f"样本{i+1}: x={x}, target={t}, pred=[{', '.join(p_formatted)}], pred_class={pred_class}, true_class={true_class}")
    print(f"Accuracy: {correct2}/{len(X)} = {100*correct2/len(X):.1f}%")

    print("\n" + "=" * 80)
    print("广义RBF网络 (Generalized RBF Network)")
    net3 = GeneralizedRBFNetwork(n_input=2, n_hidden=8, n_output=3, seed=1)
    net3.fit(X, T)
    preds3 = net3.predict(X)
    
    print(f"隐层神经元数: {net3.n_hidden}")
    print(f"Centers shape: {net3.centers.shape}")
    print(f"每个隐层神经元的Sigma: {net3.sigmas}")
    print("\nPredictions:")
    correct3 = 0
    for i, (x, t, p) in enumerate(zip(X, T, preds3)):
        pred_class = np.argmax(p)
        true_class = np.argmax(t)
        correct3 += (pred_class == true_class)
        p_formatted = [f"{val:7.4f}" for val in p]
        print(f"样本{i+1}: x={x}, target={t}, pred=[{', '.join(p_formatted)}], pred_class={pred_class}, true_class={true_class}")
    print(f"Accuracy: {correct3}/{len(X)} = {100*correct3/len(X):.1f}%")



