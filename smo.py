# coding:utf-8
import numpy as np
#import matplotlib.pyplot as plt
#import scipy.spatial.distance as dist
#from utils.misc import bar_widgets
#import logging
#from pca.pca import PCA
import torch 
#from support_vector_machine.kernels import *

#import logging

#try:
#    from sklearn.model_selection import train_test_split
#except ImportError:
#    from sklearn.cross_validation import train_test_split
#from sklearn.datasets import make_classification
#
##from pca.pca import *
#from support_vector_machine.kernels import *
#from support_vector_machine.svmModel import *
#logging.basicConfig(level=logging.DEBUG)

#import time


##线性核函数
#class LinearKernel(object):
#    def __call__(self, x, y):
#        return np.dot(x, y.T)
#
#
##多项式核函数
#class PolyKernel(object):
#    #初始化方法
#    def __init__(self, degree=2):
#        self.degree = degree
#
#    def __call__(self, x, y):
#        return np.dot(x, y.T) ** self.degree

#高斯核函数
class RBF(object):
    def __init__(self, gamma=0.1):
        self.gamma = gamma

    def __call__(self, x, y):
        #x = torch.atleast_2d(x)
        #y = torch.atleast_2d(y)
        return torch.exp(-self.gamma * torch.dist(x, y) ** 2)#.flatten()

class SVM():
    def __init__(self,trainX,trainY, C=1, kernel=None, difference=1e-3, max_iter=100):
        

        self.C = C  #正则化的参数
        self.difference = difference #用来判断是否收敛的阈值
        self.max_iter = max_iter #迭代次数的最大值

        if kernel is None:
            self.kernel = LinearKernel()  # 无核默认是线性的核
        else:
            self.kernel = kernel

        self.b = 0 # 偏置值
        self.alpha = None # 拉格朗日乘子
        self.K = None # 特征经过核函数转化的值
        self.X = trainX
        self.Y = trainY.type('torch.FloatTensor')
        self.m = trainX.shape[0]
        self.n = trainX.shape[1]
        self.K = torch.zeros((self.m))
        self.k_v = torch.zeros((self.m)) #核的新特征数组初始化
       # self.Ki = torch.zeros((self.m))
        #self.bar = progressbar.ProgressBar(widgets=bar_widgets)  # 进度条

       # for i in range(self.m):
       #     self.K[:, i] = self.kernel(self.X, self.X[i, :]) #每一行数据的特征通过核函数转化 n->m

        for i in range(self.m):
            self.K[i]=self.kernel(self.X[i],self.X[i])

        self.alpha = torch.zeros(self.m) #拉格朗日乘子初始化


    def train(self):

        for now_iter in range(self.max_iter):
            
            print (now_iter)

            alpha_prev = self.alpha.clone()
            for j in range(self.m):

                print ('1 ',j)
                
                #选择第二个优化的拉格朗日乘子
                i = self.random_index(j)
                
                print ('1.1')
                
                error_i = self.error_row(i)
                
                print ('1.2')
                
                error_j = self.error_row(j)
                
                print('2 ',j)

                #检验他们是否满足KKT条件，然后选择违反KKT条件最严重的self.alpha[j]
                if (self.Y[j] * error_j < -0.001 and self.alpha[j] < self.C) or (self.Y[j] * error_j > 0.001 and self.alpha[j] > 0):

                    print ('3 ',j)
                    
                    Kij = self.kernel(self.X[i],self.X[j])
                    
                    eta = 2.0 * Kij - self.K[i] - self.K[j]
                    
                    #eta = 2.0 * self.K[i, j] - self.K[i, i] - self.K[j, j]  #第j个要优化的拉格朗日乘子，最后需要的

                    if eta >= 0:
                        continue
                    
                    print ('4 ',j)

                    L, H = self.getBounds(i, j)
                    old_alpha_j, old_alpha_i = self.alpha[j], self.alpha[i]  #旧的拉格朗日乘子的值
                    self.alpha[j] -= (self.Y[j] * (error_i - error_j)) / eta  #self.alpha[j]的更新

                    #根据约束最后更新拉格朗日乘子self.alpha[j]，并且更新self.alpha[j]
                    self.alpha[j] = self.finalValue(self.alpha[j], H, L)
                    self.alpha[i] = self.alpha[i] + self.Y[i] * self.Y[j] * (old_alpha_j - self.alpha[j])

                    #更新偏置值b
                    b1 = self.b - error_i - self.Y[i] * (self.alpha[i] - old_alpha_j) * self.K[i] - \
                         self.Y[j] * (self.alpha[j] - old_alpha_j) * Kij#self.K[i, j]
                    b2 = self.b - error_j - self.Y[j] * (self.alpha[j] - old_alpha_j) * self.K[j] - \
                         self.Y[i] * (self.alpha[i] - old_alpha_i) * Kij#self.K[i, j]
                    if 0 < self.alpha[i] < self.C:
                        self.b = b1
                    elif 0 < self.alpha[j] < self.C:
                        self.b = b2
                    else:
                        self.b = 0.5 * (b1 + b2)
                   
            #print('3 ',j)
            #判断是否收敛
            diff = torch.norm(self.alpha - alpha_prev,p=1)
            if diff < self.difference:
                break


    #随机一个要优化的拉格朗日乘子，该乘子必须和循环里面选择的乘子不同
    def random_index(self, first_alpha):
        i = first_alpha
        while (i == first_alpha):
          i = np.random.randint(0, self.m - 1)
        return i

    #用带拉格朗日乘子表示的w代入wx+b
    def predict_row(self, X):
        
   #     print (X.shape)
        
   #     print (self.X.shape)
   
        print ("1.1.1")
        
        self.k_v = torch.zeros((self.m)) 
        
        print ("1.1.2")
        
        for i in range(self.m):
            self.k_v[i] = self.kernel(self.X[i],X)
            
        print ("1.1.3")
        
    #    k_v = self.kernel(self.X, X)
        
    #    print (k_v.shape)
        
    #    print (self.Y.shape)

        return torch.dot(self.alpha * self.Y, self.k_v) + self.b

    #预测，返回一个判断正确的index的矩阵
    def predict(self, X):
        n = X.shape[0]
        result = torch.zeros(n)
        for i in range(n):
            temp = self.predict_row(X[i, :])
            if (temp>0):
                result[i]=1
            else:
                result[i]=-1
           # result[i] = np.sign()) #正的返回1，负的返回-1
        return result

    #预测的值减真实的Y
    def error_row(self, i):

        return self.predict_row(self.X[i]) - self.Y[i]

    #得到self.alpha[j]的范围约束
    def getBounds(self,i,j):

        if self.Y[i] != self.Y[j]:
            L = max(0, self.alpha[j] - self.alpha[i])
            H = min(self.C, self.C - self.alpha[i] + self.alpha[j])
        else:
            L = max(0, self.alpha[i] + self.alpha[j] - self.C)
            H = min(self.C, self.alpha[i] + self.alpha[j])
        return L, H


    #根据self.alpha[i]的范围约束获得最终的值
    def finalValue(self,alpha,H,L):

        if alpha > H:
            alpha = H
        elif alpha < L:
            alpha = L

        return alpha

#返回准确率
def accuracy(actual, predicted):
    return 1.0 - sum(actual != predicted) / float(actual.shape[0])



#引用pca里面的模块,把数据用二维数据图表示
#def plot_in_2d(X, y=None, title=None, accuracy=None, legend_labels=None):
#
#    cmap = plt.get_cmap('viridis')
#    X_transformed = PCA().transform(X, 2)
#
#    x1 = X_transformed[:, 0]
#    x2 = X_transformed[:, 1]
#    class_distr = []
#    y = np.array(y).astype(int)
#
#    colors = [cmap(i) for i in np.linspace(0, 1, len(np.unique(y)))]
#
#    # Plot the different class distributions
#    for i, l in enumerate(np.unique(y)):
#        _x1 = x1[y == l]
#        _x2 = x2[y == l]
#        _y = y[y == l]
#        class_distr.append(plt.scatter(_x1, _x2, color=colors[i]))
#
#    # Plot legend
#    if not legend_labels is None:
#        plt.legend(class_distr, legend_labels, loc=1)
#
#    # Plot title
#    if title:
#        if accuracy:
#            perc = 100 * accuracy
#            plt.suptitle(title)
#            plt.title("Accuracy: %.1f%%" % perc, fontsize=10)
#        else:
#            plt.title(title)
#
#    # Axis labels
#    plt.xlabel('Principal Component 1')
#    plt.ylabel('Principal Component 2')
#
#    plt.show()
#    
#def run(gamma=0.1,X_train=None,y_train=None,max_iter=500,C=0.6):
#    start = time.clock()
##    X, y = make_classification(n_samples=1200, n_features=10, n_informative=5,
##                               random_state=1111, n_classes=2, class_sep=1.75, )
##    # y的标签取值{0,1} 变成 {-1, 1}
##    y = (y * 2) - 1
##    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
##                                                        random_state=1111)
#
#
#    #这里是用高斯核，可以用线性核函数和多项式核函数
#    kernel = RBF(gamma=gamma)
#    model = SVM(X_train,y_train,max_iter=max_iter, kernel=kernel, C=C)
#    model.train()
#
# #   predictions = model.predict(X_test)
#
# #   accuracyRate = accuracy(y_test, predictions)
#
# #   print('Classification accuracy (%s): %s'
# #         % (kernel, accuracyRate))
#
#
#    #原来的数据的呈现
#    #plot_in_2d(X_test, y_test, title="Support Vector Machine", accuracy=accuracyRate)
#
#    #分类的效果
##    plot_in_2d(X_test, predictions, title="Support Vector Machine", accuracy=accuracyRate)
#
#    end = time.clock()
#    print("read: %f s" % (end - start))