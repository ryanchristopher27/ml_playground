import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

class GrowingNeuralGas:
    def __init__(self, input_dim, max_nodes, max_age, epsilon_b, epsilon_n, alpha, beta, lambda_, initial_nodes=2):
        self.input_dim = input_dim
        self.max_nodes = max_nodes
        self.max_age = max_age
        self.epsilon_b = epsilon_b
        self.epsilon_n = epsilon_n
        self.alpha = alpha
        self.beta = beta
        self.lambda_ = lambda_
        
        # Initialize nodes
        self.nodes = np.random.rand(initial_nodes, input_dim)
        self.errors = np.zeros(initial_nodes)
        self.ages = np.zeros((initial_nodes, initial_nodes))
        self.connections = np.zeros((initial_nodes, initial_nodes))
        
        self.num_nodes = initial_nodes

    def find_nearest(self, x):
        distances = np.linalg.norm(self.nodes - x, axis=1)
        nearest = np.argmin(distances)
        second_nearest = np.argpartition(distances, 1)[1]
        return nearest, second_nearest

    def update_nodes(self, x, nearest, second_nearest):
        self.errors[nearest] += np.linalg.norm(self.nodes[nearest] - x)**2
        self.nodes[nearest] += self.epsilon_b * (x - self.nodes[nearest])
        for i in range(self.num_nodes):
            if self.connections[nearest, i] > 0:
                self.ages[nearest, i] += 1
                self.nodes[i] += self.epsilon_n * (x - self.nodes[i])

    def add_node(self):
        q = np.argmax(self.errors)
        f = np.argmax(self.connections[q])
        new_node = (self.nodes[q] + self.nodes[f]) / 2.0
        self.nodes = np.vstack([self.nodes, new_node])
        self.errors = np.append(self.errors, 0)
        self.connections = np.pad(self.connections, ((0, 1), (0, 1)), 'constant')
        self.ages = np.pad(self.ages, ((0, 1), (0, 1)), 'constant')
        
        self.connections[q, f] = self.connections[f, q] = 0
        self.connections[q, self.num_nodes] = self.connections[self.num_nodes, q] = 1
        self.connections[f, self.num_nodes] = self.connections[self.num_nodes, f] = 1
        
        self.errors[q] *= self.alpha
        self.errors[f] *= self.alpha
        
        self.num_nodes += 1

    def remove_old_connections(self):
        self.connections[self.ages > self.max_age] = 0
        self.ages[self.ages > self.max_age] = 0
        to_remove = np.where(np.sum(self.connections, axis=0) == 0)[0]
        self.nodes = np.delete(self.nodes, to_remove, axis=0)
        self.errors = np.delete(self.errors, to_remove)
        self.connections = np.delete(self.connections, to_remove, axis=0)
        self.connections = np.delete(self.connections, to_remove, axis=1)
        self.ages = np.delete(self.ages, to_remove, axis=0)
        self.ages = np.delete(self.ages, to_remove, axis=1)
        self.num_nodes -= len(to_remove)

    def train(self, data, epochs, interval=10):
        fig, ax = plt.subplots()
        sc = ax.scatter(self.nodes[:, 0], self.nodes[:, 1])
        lines = []

        def update(frame):
            ax.clear()
            for _ in range(interval):
                for x in data:
                    nearest, second_nearest = self.find_nearest(x)
                    self.update_nodes(x, nearest, second_nearest)
                    self.ages[nearest, :] += 1
                    self.ages[:, nearest] += 1
                    self.ages[nearest, second_nearest] = self.ages[second_nearest, nearest] = 0
                    self.connections[nearest, second_nearest] = self.connections[second_nearest, nearest] = 1

                    if frame % self.lambda_ == 0 and self.num_nodes < self.max_nodes:
                        self.add_node()
                    self.errors *= self.beta
                    self.remove_old_connections()

            ax.scatter(self.nodes[:, 0], self.nodes[:, 1])
            for i in range(self.num_nodes):
                for j in range(i + 1, self.num_nodes):
                    if self.connections[i, j] > 0:
                        line, = ax.plot([self.nodes[i, 0], self.nodes[j, 0]], [self.nodes[i, 1], self.nodes[j, 1]], 'k-')
                        lines.append(line)

        ani = FuncAnimation(fig, update, frames=epochs // interval, repeat=False)
        plt.show()

    def best_number_of_clusters(self, data):
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)
        scores = []
        for k in range(2, self.max_nodes + 1):
            self.nodes = np.random.rand(k, self.input_dim)
            self.num_nodes = k
            self.errors = np.zeros(k)
            self.ages = np.zeros((k, k))
            self.connections = np.zeros((k, k))
            for _ in range(100):
                for x in data_scaled:
                    nearest, second_nearest = self.find_nearest(x)
                    self.update_nodes(x, nearest, second_nearest)
                    self.ages[nearest, :] += 1
                    self.ages[:, nearest] += 1
                    self.ages[nearest, second_nearest] = self.ages[second_nearest, nearest] = 0
                    self.connections[nearest, second_nearest] = self.connections[second_nearest, nearest] = 1
                self.errors *= self.beta
                self.remove_old_connections()
            labels = np.array([self.find_nearest(x)[0] for x in data_scaled])
            score = silhouette_score(data_scaled, labels)
            scores.append(score)
        best_k = np.argmax(scores) + 2
        return best_k

# Simulate streaming stock data
def generate_stock_data(num_stocks=100, num_packets=10):
    symbols = [f'Stock{i}' for i in range(num_stocks)]
    data_packets = []
    for _ in range(num_packets):
        prices = np.random.rand(num_stocks, 1) * 100
        feature1 = np.random.rand(num_stocks, 1) * 10
        df = pd.DataFrame(np.hstack((prices, feature1)), columns=['price', 'feature1'])
        df['symbol'] = symbols
        data_packets.append(df)
    return data_packets

# Example usage
input_dim = 2  # Dimensionality of the input data
max_nodes = 20  # Maximum number of nodes
max_age = 50  # Maximum age of edges
epsilon_b = 0.05  # Learning rate for the nearest node
epsilon_n = 0.006  # Learning rate for the neighbors
alpha = 0.5  # Error reduction factor
beta = 0.995  # Error decay factor
lambda_ = 100  # Frequency of adding new nodes

gng = GrowingNeuralGas(input_dim, max_nodes, max_age, epsilon_b, epsilon_n, alpha, beta, lambda_)

# Generate and process streaming stock data
data_packets = generate_stock_data()
for packet in data_packets:
    data = packet[['price', 'feature1']].values
    gng.train(data, epochs=100)

# Determine the best number of clusters
all_data = np.vstack([packet[['price', 'feature1']].values for packet in data_packets])
best_k = gng.best_number_of_clusters(all_data)
print(f'The best number of clusters is: {best_k}')
