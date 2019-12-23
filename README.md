## C++ 实现的机器学习算法 函数说明

[python版](https://github.com/dmrasf/mat-py)

#### Net

|          函数         |                作用               |
|:---------------------:|:---------------------------------:|
|       Net net(n)      |      建一个输入为n的神经网络      |
|  net.add_lay(n, func) | n：下一层的节点数  func：激活函数 |
| net.predict(MatrixXd) |              预测输出             |

#### Clustering

|          函数         |            作用            |
|:---------------------:|:--------------------------:|
|  Clustering clu(m, n) | m：要分的类别  n：输入维度 |
| clu.predict(MatrixXd) |          预测输出          |

#### Svm

|          函数         |      作用      |
|:---------------------:|:--------------:|
|  Svm svm(c, tol, det) | 初始化训练参数 |
| svm.predict(MatrixXd) |      预测      |

#### Train

|            函数           |            作用            |
|:-------------------------:|:--------------------------:|
|      Train tra(x, y)      |       初始化训练数据       |
|     tra.train(Net&, n)    | 训练Net  使用累积BP训练n次 |
|   tra.train_BP(Net&, n)   |      训练Net  训练n次      |
| tra.train(Clustering&, n) |     训练Clustering n次     |
|      tra.train(Svm&)      |     计算 a  和支持向量     |


#### 数字识别需要的函数

|        函数        |        作用        |
|:------------------:|:------------------:|
| date_x(x, path, n) | 读取训练集或测试集 |
|  read_bmp(path, x) |      读取图片      |

--------

### 试例

```C++
// 使用不同的算法解决异或问题
int main(){
    //定义训练数据
    MatrixXd x_train(2, 4), y_train(1, 4);
    x_train << 0, 0, 1, 1,
               1, 0, 1, 0;
    y_train << 1, 0, 0, 1;

    //net
    Net net(2);
    net.add_lay(2, 'sigmoid');
    net.add_lay(1, 'sigmoid');

    //clu
    Svm svm;

    //train
    Train tra(x_train, y_train);
    tra.train(net, 10);
    tra.train(svm);

    //predict
    cout << "net:" << net.predict(x_train) << endl;
    cout <<< "svm" << svm.predict(x_train) << endl;

    return 0;
}
```
--------

###### *需要用到的矩阵库*

[Eigen](http://eigen.tuxfamily.org/index.php?title=Main_Page)


