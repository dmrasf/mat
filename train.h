#ifndef TRAIN_H
#define TRAIN_H

#include "net.h"


class Train
{
public:
	Train();
	//放入训练集 
	Train(const MatrixXd&, const MatrixXd&);
	//训练函数 
	bool train(Net&, int);
	//计算用于训练的数据 
	bool calculate(Net&); 
	~Train();
private:
	MatrixXd x_train;
	MatrixXd y_train;
	//保存每层的输出 
	vector<MatrixXd> z;
	//保存每层的输入 
	vector<MatrixXd> e;
};

#endif
