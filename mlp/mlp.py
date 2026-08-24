import numpy as np

def sigmoide(x: np.ndarray):
    return 1 / (1 + np.exp(-x))

def sigmoide_prime(x: np.ndarray):
    return x * (1 - x)

def relu(x: np.ndarray):
    return np.maximum(0, x)

def relu_prime(x: np.ndarray):
    return np.where(x >= 0, 1, 0)

def mse(predicted: np.ndarray, expected: np.ndarray):
    return np.mean((expected - predicted)**2)

def mse_prime(predicted: np.ndarray, expected: np.ndarray):
    return 2 * (predicted - expected) / predicted.shape[0]

class Layer:
    def __init__(self, n_in: int, n_out: int):
        self.n_in, self.n_out = n_in, n_out

        self.b = np.random.randn(n_out)
        self.w = np.random.randn(n_out, n_in)

        self.last_activation = None
        self.last_entries = None

        self.activation = relu
        self.activation_prime = relu_prime

    def set_activation(self, activation, activation_prime):
        self.activation = activation
        self.activation_prime = activation_prime

    def forward_prop(self, entries: np.ndarray) -> np.ndarray:
        self.last_entries = entries
        self.last_activation = self.activation(np.dot(entries,  self.w.T) + self.b)
        return self.last_activation

    def back_prop(self, delta: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]: #return (dw, db, delta)
        delta = delta * self.activation_prime(self.last_activation)
        dw = delta.T @ self.last_entries
        db = np.sum(delta, axis=0)
        delta = delta @ self.w
        return (dw, db, delta)

class MLP:
    def __init__(self, layer_sizes: list[int]):
        self.nb_layers = len(layer_sizes) - 1
        self.layers = []
        for i in range(1, len(layer_sizes)):
            self.layers.append(Layer(layer_sizes[i-1], layer_sizes[i]))

        self.cost_func = mse
        self.cost_func_prime = mse_prime

    def predict(self, entries: np.ndarray) -> np.ndarray: #launch forward prop and return result
        for i in range(self.nb_layers):
            entries = self.layers[i].forward_prop(entries)
        return entries 

    def gradient_descent(self, predicted: np.ndarray, expected: np.ndarray, learning_rate: float): #launch back prop and apply correction
        delta = self.cost_func_prime(predicted, expected)
        for i in reversed(range(self.nb_layers)):
            (dw, db, delta) = self.layers[i].back_prop(delta)
            self.layers[i].w -= learning_rate * dw
            self.layers[i].b -= learning_rate * db

    def train(self, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray= None, y_test: np.ndarray= None, nb_iteration: int= 1000, learning_rate: float= 1.)\
          -> tuple[list[float], list[float]] | list[float]: #return train_losses and test_losses or only train_losses
        train_losses = []
        if X_test is not None:
            test_losses = []
        
        for i in range(nb_iteration):
            pred = self.predict(X_train)
            train_losses.append(self.cost_func(pred, y_train))
            self.gradient_descent(pred, y_train, learning_rate)

            if X_test is not None:
                test_pred = self.predict(X_test)
                test_losses.append(self.cost_func(test_pred, y_test))

        if X_test is not None:
            return (train_losses, test_losses)
        return train_losses