from PIL import Image
from os import listdir
import numpy as np
import pickle as pk


def image_to_matrix(file_path):
    matrix = np.zeros(28 * 28)
    image = Image.open(file_path).convert('RGB')
    for i in range(28):
        for j in range(28):
            matrix[28 * i + j] = int(image.getpixel((i, j))[0] / 255)
    return matrix


def load_image(filename, number):
    X = []
    Y = []
    y = np.zeros(12)
    y[number] = 1
    file_list_name = listdir(filename)
    for i in range(len(file_list_name)):
        X.append(image_to_matrix(filename + "/" + file_list_name[i]))
        Y.append(y)
    return X, Y


def load_images(filename):
    X = []
    Y = []
    file_list_name = listdir(filename)
    for i in range(len(file_list_name)):
        x, y = load_image(filename + "/" + file_list_name[i], int(file_list_name[i])-1)
        X += x
        Y += y
    return np.array(X), np.array(Y)

def load_test():
    X = []
    Y = []
    for i in range(1,1801):
        name = "./111/test/"+str(i)+".bmp"
        matrix = image_to_matrix(name)
        X.append(matrix)
        Y.append(i)
    return np.array(X),np.array(Y)


def dump(input, filename):
    fw = open(filename, "wb")
    pk.dump(input, fw)
    fw.close()


X_data,Y_data = load_images("./train")

dump(X_data,"./x_train.txt")
dump(Y_data,"./y_train.txt")