import numpy as np


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

            # gradient descent updates (batch)
            self.w2 -= eta * (layer2.T.dot(delta3)) 
            self.w1 -= eta * (layer1.T.dot(delta2)) 

            if verbose and epoch % print_every == 0:
                mse = np.mean(0.5 * np.square(T - out))
                print(f"Epoch {epoch}, MSE: {mse:.6f}")

    def predict(self, X):
        _, _, out = self.forward(np.asarray(X, dtype=float))
        return out


if __name__ == '__main__':

    X = np.array([[0.75, 1.0], [0.5, 0.75], [0.25, 0.0], [0.5, 0.0],[0.0, 0.0], [1.0, 0.75], [1.0, 1.0], [0.5, 0.25], [0.75, 0.5]])
    T = np.array([[1, -1, -1], [1, -1, -1], [1, -1, -1], [-1, 1, -1], [-1, 1, -1], [-1, 1, -1], [-1, -1, 1], [-1, -1, 1], [-1, -1, 1]])

    # instantiate network matching target range (here targets are -1/1 -> use tanh)
    net = BPNetwork(n_input=2, n_hidden=4, n_output=3, seed=1, output_activation='tanh')
    print('Initial weights (w1):\n', net.w1)
    print('Initial weights (w2):\n', net.w2)

    net.train(X, T, epochs=10000, eta=0.5, verbose=True, print_every=2000)

    preds = net.predict(X)
    print('\nFinal predictions:')
    for x, t, p in zip(X, T, preds):
        p_formatted = [f"{val:.4f}" for val in p]
        print(f"x={x}, target={t}, pred={p_formatted}")