#ifndef TRAIN_H
#define TRAIN_H
#include "net.h"
#include "rbf.h"
#include "clustering.h"
#include "svm.h" 

class Train
{
public:
	Train();
	Train(double); 
	void get_new(const MatrixXd&, const MatrixXd&);
	//放入训练集 
	Train(const MatrixXd&, const MatrixXd&);
	//训练函数 
	//累积BP 
	bool train(Net&, int);
	//单个训练 
	bool train_BP(Net&, int);
	bool train(Rbf&, int);
	bool train(Clustering&, int);
	bool train(Svm&);
	//计算用于训练的数据 
	bool calculate(Net&);
	MatrixXd get_d(Net&, int, const MatrixXd&);
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
	double rate = 0.3;
};

#endif
