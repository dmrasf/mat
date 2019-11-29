#ifndef TRAIN_H
#define TRAIN_H
#include "net.h"
#include "rbf.h"

class Train
{
public:
	Train();
	Train(double); 
	void get_new(const MatrixXd&, const MatrixXd&);
	//放入训练集 
	Train(const MatrixXd&, const MatrixXd&);
	//训练函数 
	bool train(Net&, int);
	bool train(Rbf&, int);
	//计算用于训练的数据 
	bool calculate(Net&);
	//test
	void show_cal();
	~Train();
private:
	MatrixXd x_train;
	MatrixXd y_train;
	//保存每层的输出 
	vector<MatrixXd> z;
	//保存每层的输入 
	vector<MatrixXd> e;
	double rate;
};

#endif
