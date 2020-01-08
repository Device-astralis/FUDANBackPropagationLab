import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


class BackPropagation:
    def __init__(self, hidden_layer=None, hidden_activation='sigmoid', output_activation='sigmoid', epoch_number=1000, learning_rate=0.1):
        self.input_layer = None  # 输入层神经元数目
        self.hidden_layer = hidden_layer  # 隐藏层神经元数目
        self.output_layer = None # 输出结果数量，也就是输出层神经元数量
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.epoch_number = epoch_number
        self.learning_rate = learning_rate  # 学习率，是运用梯度下降算法时，W系数的变化率

        self.weight = [] # 每两层之间的权值矩阵
        self.bias = [] # 每两层之间的偏置矩阵
        self.N = None    # 样本数量

    def init_param(self, X_data, Y_data,layer):
        # 初始化
        self.N = X_data.shape[0]
        self.input_layer = X_data.shape[1]
        self.output_layer = Y_data.shape[1]
        self.hidden_layer = layer[1:len(layer)-1]
        
        for i in range(len(self.hidden_layer)):
            if i == 0:
                self.weight.append(np.random.randn (self.input_layer, self.hidden_layer[i]))
                self.bias.append(np.random.randn(self.hidden_layer[i]))
            if i < len(self.hidden_layer)-1:
                self.weight.append(np.random.randn(self.hidden_layer[i],self.hidden_layer[i+1]))
                self.bias.append(np.random.randn(self.hidden_layer[i+1]))
            if i == len(self.hidden_layer)-1:
                self.weight.append(np.random.randn(self.hidden_layer[i], self.output_layer))
                self.bias.append(np.random.randn(self.output_layer))
        return X_data, Y_data

    @staticmethod
    def get_activation_function(name,x):
        if name == 'sigmoid':
            return 1.0 / (1 + np.exp(-x))
        elif name == 'linear':
            return x

    def get_activation_derivation(self,name,x):
        if name == 'sigmoid':
            val = self.get_activation_function('sigmoid',x)
            return val * (1 - val)
        elif name == 'linear':
            return np.ones_like(x)

    def forward(self, X_data):
        #a是状态值，z是激活值，好像弄反了，懒得改。。
        cell_in = []
        cell_out = [X_data]
        # 前向传播
        for i in range(len(self.hidden_layer)):
            right_output=[]
            if i == 0:
                left_input = np.dot(X_data,self.weight[0]) + self.bias[0]
                right_output = self.get_activation_function(self.hidden_activation,left_input)
                cell_in.append(left_input)
                cell_out.append(right_output)
            left_input = np.dot(cell_out[i + 1], self.weight[i + 1]) + self.bias[i + 1]
            if i < len(self.hidden_layer)-1:
                right_output = self.get_activation_function(self.hidden_activation,left_input)
            if i == len(self.hidden_layer)-1:
                right_output = self.get_activation_function(self.output_activation,left_input)
            cell_in.append(left_input)
            cell_out.append(right_output)
        #存储从隐藏层的第一层开始到输出层为止，每一层的z和a
        return cell_in,cell_out

    def train(self, X_data, Y_data,layer):
        # 训练的函数
        X_data, Y_data = self.init_param(X_data, Y_data,layer)
        # 初始化
        for step in tqdm(range(self.epoch_number)):
            # 前向传播
            cell_in_list,cell_out_list = self.forward(X_data)#n+1,n+2
            # # 误差反向传播，依据权值逐层计算当层误差
            error_hidden=[]
            for i in range(len(self.hidden_layer),-1,-1):#算上隐藏层（n层），一共有n+1个权重和偏置需要进行调整
                if i == len(self.hidden_layer):
                    delta_error = (cell_out_list[i+1]-Y_data)*self.get_activation_derivation(self.output_activation,cell_in_list[i])
                    error_hidden = np.dot(delta_error,self.weight[i].T)
                    self.bias[i] -= np.sum(self.learning_rate*delta_error,axis=0) / self.N
                    self.weight[i] -= self.learning_rate * np.dot(cell_out_list[i].T,delta_error)/self.N
                else:
                    delta_error = error_hidden*self.get_activation_derivation(self.hidden_activation,cell_in_list[i])
                    error_hidden = np.dot(delta_error,self.weight[i].T)
                    self.bias[i] -= np.sum(self.learning_rate*delta_error,axis=0)/self.N
                    self.weight[i] -= self.learning_rate*np.dot(cell_out_list[i].T,delta_error)/self.N

        return

    def predict(self, X):
        # 预测
        final_a,final_z = self.forward(X)
        return final_z[-1]


if __name__ == '__main__':
    N = 1000
    layer = [1,10,10,1]
    X_data = np.random.rand(N)*np.pi*2-np.pi
    X_data = np.transpose([X_data])
    Y_data = np.sin(X_data)
    BackPropagation = BackPropagation(output_activation='linear', epoch_number=100000, learning_rate=0.1)
    BackPropagation.train(X_data, Y_data,layer)

    X_data = np.random.rand(1000)*np.pi*2-np.pi
    X_data = np.transpose([X_data])
    Y_data = np.sin(X_data)
    plt.scatter(X_data, Y_data)
    pred = BackPropagation.predict(X_data)
    print(np.sum(abs(pred - Y_data)) / 1000)
    plt.scatter(X_data, pred, color='r')
    plt.show()