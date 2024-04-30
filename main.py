import numpy as np
import matplotlib.pyplot as plt

from tabulate import tabulate


# Функция активации
def activation(net):
    return (1 - np.exp(-net)) / (1 + np.exp(-net))

# Производная функции активации
def derivative_activation_function(net):
    return (1 - activation(net)**2)/2

class Layer:
    def __init__(self, neurons):
        self.neurons = neurons

class NN:
    def __init__(self, input_size, layers, epsilon=0.001, norma=1):
        self.input_size = input_size  # Размер входных данных
        self.layers = layers         # Количество нейронов в слое
        self.epsilon = epsilon        # Коэффициент для инициализации весов
        self.norma = norma            # Норма для инициализации весов

        # Инициализация весов
        self.weights = self.init_weights()
        # print(f"Инициализация весов: {self.weights}")

        # Инициализация Промежуточные значения
        self.nets = None  # сумматоры
        self.outs = None  # выходы

    def init_weights(self):
        weights = []
        prev_size = self.input_size  # size предыдущий слой = size входные данные
        for layer in self.layers:
            weights.append(np.random.rand(prev_size + 1, layer.neurons))
            prev_size = layer.neurons  # Обновляем размер предыдущего слоя для следующего нейрона
            print(f"Инициализация весов: {weights}")
        return weights

    # Обратное распространение ошибки для обновления весов (error: ошибка на выходе сети)
    def backpropagation_func(self, error):
        derivative = derivative_activation_function(self.nets[-1])
        delta = [derivative * error]  # дельта для последнего слоя

        # Обратное распространение ошибки по всем слоям сети
        for i, nets in enumerate(reversed(self.nets[:-1]), 1):
            derivative = derivative_activation_function(nets)
            delta.append(derivative * np.dot(self.weights[-i][1:], delta[-1]))  # дельта для текущего слоя

        # Обновление весов по дельтам
        for i, weights in enumerate(reversed(self.weights)):
            weights[0] += self.norma * delta[i]  # Обновление веса 0 индекса
            weights[1:] += self.norma * np.outer(self.outs[-i-2], delta[i])  # Обновление

    # Предсказание выхода сети для входных данных
    def network_output_prediction_input_data(self, x_input):
        nets, outs = [], [x_input]
        out = x_input  # первый слой = входное данное

        for layer, weights in zip(self.layers, self.weights):
            net = np.dot(out, weights[1:]) + weights[0]
            nets.append(net)
            out = activation(net)
            outs.append(out)

        self.nets, self.outs = nets, outs  # Сохраняем для дальше
        return out  #  предсказание

    # Обучение нейронной сети
    def lerning_func(self,
            x_train, # обучающие данные
            t_train # целевые значения
            ):

        k = -1
        error_mse = np.inf  # ошибка MSE
        error_mse_arr = []

        while error_mse > self.epsilon:
            k += 1
            y = self.network_output_prediction_input_data(x_train)  # предсказанные значения сети
            error = t_train - y
            error_mse = np.sqrt(np.sum(error**2))  # MSE
            error_mse_arr.append(error_mse)

            data = []
            data.append(["Номер эпохи", "Ошибка E(k)", "Выходной вектор y"])
            data.append([k, error_mse.round(4), y.round(4)])

            for i, w in enumerate(self.weights, 1):
                if i != 1:
                    data.append([f"Веса скрытого слоя", "\n".join(map(str, w.round(4))), ""])
                else:
                    data.append([f"Веса выходного слоя", "\n".join(map(str, w.round(4))), ""])
            print(tabulate(data, tablefmt="plain", headers="firstrow"))
            print("\n")
            if error_mse > self.epsilon:
                self.backpropagation_func(error)  # обновление. обратное распространение ошибки

        print(f"t: {t_train}")

        plt.plot(range(len(error_mse_arr)), error_mse_arr)
        plt.title('Параметры НС на последовательных эпохах'), plt.xlabel('Номер эпохи k'), plt.ylabel('Ошибка E(k)')
        plt.grid(), plt.show()

x = [1, 3]
t = [0.1]

'''
Архитектура 1-2-1 : нейронная сеть состоит из трех слоев
1. Входной слой с одним нейроном: нейрон принимает входные данные
2. Скрытый слой с двумя нейронами: вычисление внутренних представлений данных
3. Выходной слой с одним нейроном: Этот нейрон принимает выходные данные от скрытого слоя и генерирует окончательный результат
'''

# (1-2-1) два слоя с количеством нейронов 2 и 1

nn = NN(input_size=len(x), layers=[Layer(neurons=1), Layer(neurons=2)])
nn.lerning_func(x_train=x, t_train=t)
