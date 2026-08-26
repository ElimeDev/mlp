import numpy as np

def sigmoide(x: np.ndarray):
    return 1 / (1 + np.exp(-x))

def sigmoide_prime(x: np.ndarray):
    return x * (1 - x)

def relu(x: np.ndarray):
    return np.maximum(0, x)

def relu_prime(x: np.ndarray):
    return np.where(x > 0, 1, 0)

def mse(predicted: np.ndarray, expected: np.ndarray):
    return np.mean((expected - predicted)**2)

def mse_prime(predicted: np.ndarray, expected: np.ndarray):
    return 2 * (predicted - expected) / predicted.shape[0]

def cross_entropy(predicted: np.ndarray, expected: np.ndarray):
    predicted = np.clip(predicted, np.finfo(float).eps, 1 - np.finfo(float).eps)
    return -np.mean(expected * np.log(predicted) + (1 - expected) * np.log(1 - predicted))

def cross_entropy_prime(predicted: np.ndarray, expected: np.ndarray):
    predicted = np.clip(predicted, np.finfo(float).eps, 1 - np.finfo(float).eps)
    return (predicted - expected) / (predicted * (1 - predicted)) / expected.shape[0]

class Layer:
    def __init__(self, n_in: int, n_out: int):
        self.n_in, self.n_out = n_in, n_out

        self.b = np.random.randn(n_out)
        self.w = np.random.randn(n_out, n_in)

        self.last_activation = None
        self.last_entries = None

        self.activation = sigmoide
        self.activation_prime = sigmoide_prime

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

        #only train_losses is computed
        self.last_training_data = {
            "train_losses" : None, 
            "test_losses" : None, 
            "train_accuracy" : None, 
            "test_accuracy" : None
        }

    def get_last_training_data(self):
        return self.last_training_data

    def set_cost_func(self, cost_func, cost_func_prime):
        self.cost_func = cost_func
        self.cost_func_prime = cost_func_prime

    def set_hidden_layers_activation(self, activation, activation_prime):
        for layer in self.layers[:-1]:
            layer.set_activation(activation, activation_prime)

    def set_output_layers_activation(self, activation, activation_prime):
        self.layers[-1].set_activation(activation, activation_prime)

    def predict(self, entries: np.ndarray) -> np.ndarray: #launch forward prop and return result
        for i in range(self.nb_layers):
            entries = self.layers[i].forward_prop(entries)
        return entries 

    def gradient_descent(self, predicted: np.ndarray, expected: np.ndarray, learning_rate: float, lambda_= 0.0, training_data_size= 0): #launch back prop and apply correction
        delta = self.cost_func_prime(predicted, expected)
        for i in reversed(range(self.nb_layers)):
            (dw, db, delta) = self.layers[i].back_prop(delta)
            self.layers[i].w = (1 - learning_rate * (lambda_ / training_data_size)) * self.layers[i].w - learning_rate * dw
            self.layers[i].b -= learning_rate * db

    def train(self, X_train, y_train, epochs= 1000, learning_rate= 1., mini_batch_size= False, lambda_= False):
        train_losses = []

        if not mini_batch_size:
            mini_batch_size = len(X_train)

        for i in range(epochs):
            indices = np.random.permutation(len(X_train))
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]

            batch_losses = []
            for start in range(0, len(X_shuffled), mini_batch_size):
                X_batch = X_shuffled[start : start + mini_batch_size]
                y_batch = y_shuffled[start : start + mini_batch_size]

                pred = self.predict(X_batch)
                batch_losses.append(self.cost_func(pred, y_batch))
                self.gradient_descent(pred, y_batch, learning_rate, lambda_= lambda_, training_data_size= X_train.shape[0])

            train_losses.append(np.mean(batch_losses))

        self.last_training_data["train_losses"] = train_losses