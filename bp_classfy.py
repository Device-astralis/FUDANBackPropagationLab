import numpy as np
import pickle as pk
from tqdm import tqdm


class BackPropagation:
    def __init__(self, hidden_layer=None, hidden_activation='sigmoid', output_activation='sigmoid', epoch_number=1000,
                 learning_rate=0.1):
        self.input_layer = None  # 输入层神经元数目
        self.hidden_layer = hidden_layer  # 隐藏层神经元数目
        self.output_layer = None  # 输出结果数量，也就是输出层神经元数量
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.epoch_number = epoch_number
        self.learning_rate = learning_rate  # 学习率，是运用梯度下降算法时，W系数的变化率

        self.weight = []  # 每两层之间的权值矩阵
        self.bias = []  # 每两层之间的偏置矩阵
        self.N = None  # 样本数量

    def init_param(self, X_data, Y_data, layer):
        # 初始化
        if len(X_data.shape) == 1:  # 若输入数据为一维数组，则进行转置为n维数组
            X_data = np.transpose([X_data])
        self.N = X_data.shape[0]
        if len(Y_data.shape) == 1:
            Y_data = np.transpose([Y_data])
        self.input_layer = X_data.shape[1]  # 将输入层神经元数量赋值给变量
        self.output_layer = Y_data.shape[1]  # 将输出层神经元数量赋值给变量
        self.hidden_layer = layer[1:len(layer) - 1]

        for i in range(len(self.hidden_layer)):
            if i == 0:
                self.weight.append(np.random.uniform(-0.1, 0.1, (self.input_layer, self.hidden_layer[i])))
                self.bias.append(np.random.uniform(-0.1, 0.1, self.hidden_layer[i]))  # 初始化输入层和第一个隐藏层之间的权重和偏置矩阵
            if i < len(self.hidden_layer) - 1:
                self.weight.append(np.random.uniform(-0.1, 0.1, (self.hidden_layer[i], self.hidden_layer[i + 1])))
                self.bias.append(np.random.uniform(-0.1, 0.1, self.hidden_layer[i + 1]))  # 初始化第i个隐藏层和第i+1个隐藏层之间的权重和偏置矩阵
            if i == len(self.hidden_layer) - 1:
                self.weight.append(np.random.uniform(-0.1, 0.1, (self.hidden_layer[i], self.output_layer)))
                self.bias.append(np.random.uniform(-0.1, 0.1, self.output_layer))  # 初始化最后一个隐藏层和输出层之间的权重和偏置矩阵
        return X_data, Y_data

    @staticmethod
    def get_activation_function(name, x):
        # 获取相应的激励函数
        if name == 'sigmoid':
            return 1.0 / (1 + np.exp(-x))
        elif name == 'linear':
            return x
        elif name == 'softmax':
            return np.exp(x) / np.sum(np.exp(x))

    def get_activation_derivation(self, name, x):
        # 获取相应的激励函数的导数
        if name == 'sigmoid':
            val = self.get_activation_function('sigmoid', x)
            return val * (1 - val)
        elif name == 'linear':
            return np.ones_like(x)

    def get_softmax_derivation(self, y_data, cell_out_list):
        loss_function = -np.divide(y_data, cell_out_list[-1])
        softmax_dervation = np.zeros((1, self.output_layer))
        temp = np.zeros((self.output_layer, self.output_layer))
        for i in range(self.output_layer):
            for j in range(self.output_layer):
                if i == j:
                    temp[i][j] = cell_out_list[-1][0][i] * (1 - cell_out_list[-1][0][i])
                else:
                    temp[i][j] = cell_out_list[-1][0][i] * (-cell_out_list[-1][0][j])
        softmax_dervation[0] = np.dot(loss_function[0], temp)
        return softmax_dervation

    def forward(self, X_data):
        # a是状态值，z是激活值，好像弄反了，懒得改。。
        cell_in = []
        cell_out = [X_data]
        # 前向传播
        for i in range(len(self.hidden_layer)):
            right_output = []
            if i == 0:
                left_input = np.dot(X_data, self.weight[0]) + self.bias[0]
                right_output = self.get_activation_function(self.hidden_activation, left_input)
                cell_in.append(left_input)
                cell_out.append(right_output)
            left_input = np.dot(cell_out[i + 1], self.weight[i + 1]) + self.bias[i + 1]
            if i < len(self.hidden_layer) - 1:
                right_output = self.get_activation_function(self.hidden_activation, left_input)
            if i == len(self.hidden_layer) - 1:
                right_output = self.get_activation_function(self.output_activation, left_input)
            cell_in.append(left_input)
            cell_out.append(right_output)
        # 存储从隐藏层的第一层开始到输出层为止，每一层的z和a
        return cell_in, cell_out

    def train(self, X_data, Y_data, layer):
        old = 0
        # 训练的函数
        X_data1 = load('./x_train.txt')
        Y_data1 = load('./y_train.txt')
        X_data, Y_data = self.init_param(X_data, Y_data, layer)
        # 打乱训练序列进行训练
        for step in tqdm(range(self.epoch_number)):
            m = []
            for i in range(len(X_data)):
                m.append(i)
            np.random.shuffle(m)
            for i in range(len(m)):
                x_data = np.reshape(X_data[m[i]], (1, 784))
                y_data = np.reshape(Y_data[m[i]], (1, 12))
                # 前向传播
                cell_in_list, cell_out_list = self.forward(x_data)  # n+1,n+2
                # # 误差反向传播，依据权值逐层计算当层误差
                error_hidden = []
                for i in range(len(self.hidden_layer), -1, -1):  # 算上隐藏层（n层），一共有n+1个权重和偏置需要进行调整
                    if i == len(self.hidden_layer):
                        delta_error = self.get_softmax_derivation(y_data, cell_out_list)
                        error_hidden = np.dot(delta_error, self.weight[i].T)
                        self.bias[i] -= np.sum(self.learning_rate * delta_error, axis=0)
                        self.weight[i] -= self.learning_rate * np.dot(cell_out_list[i].T, delta_error)
                    else:
                        delta_error = error_hidden * self.get_activation_derivation(self.hidden_activation,
                                                                                    cell_in_list[i])
                        error_hidden = np.dot(delta_error, self.weight[i].T)
                        self.bias[i] -= np.sum(self.learning_rate * delta_error, axis=0)
                        self.weight[i] -= self.learning_rate * np.dot(cell_out_list[i].T, delta_error)

            predict = self.predict(X_data1)
            count = 0
            for i in range(len(X_data1)):
                predict_type = np.argmax(predict[i])
                real_type = np.argmax(Y_data1[i])
                if predict_type == real_type:
                    count += 1
            # if (old > count):
            #     self.learning_rate *= 0.9
            # else :
            #     self.learning_rate *= 1.01
            # old = count
            print("step: %d" % step)
            print("总个数：%d,正确个数：%d" % (len(X_data1), count))
            print("正确率：%f" % (count / len(X_data1)))

    def predict(self, X):
        final_a, final_z = self.forward(X)
        return final_z[-1]


def load(file):
    f = open(file, "rb")
    return pk.load(f)


def dump(input, file):
    f = open(file, "wb")
    pk.dump(input, f)
    f.close()


def test(X, Y, BackPropagation, step):
    final_z = BackPropagation.predict(X)
    num = 0
    prediction = []
    for i in range(len(X)):
        result = np.argmax(Y[i])
        pre = np.argmax(final_z[i])+1
        prediction.append(pre)
        if pre == result:
            num += 1
    np.savetxt("./pred.txt", np.array(prediction), fmt="%d")
    print("步数: %d" % step)
    print("预测准确率：%f" % (num / len(X)))


if __name__ == '__main__':
    layer = [784, 80, 80, 12]

    X_data = load('./x_test.txt')
    Y_data = load('./y_test.txt')
    BackPropagation = BackPropagation(hidden_activation='sigmoid', output_activation='softmax', epoch_number=60,
                                      learning_rate=0.01)
    BackPropagation.init_param(X_data,Y_data,layer)
    # BackPropagation.train(X_data, Y_data, layer)
    # dump(BackPropagation.weight, "./BackPropagation_weightbs.txt")
    # dump(BackPropagation.bias, "./BackPropagation_biasbs.txt")
    X_data = load('./x_test.txt')
    Y_data = load('./y_test.txt')
    BackPropagation.weight = load("./BackPropagation_weightbs.txt")
    BackPropagation.bias = load("./BackPropagation_biasbs.txt")

    test(X_data, Y_data, BackPropagation, 1000)