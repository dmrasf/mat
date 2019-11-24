#ifndef TRAIN_H
#define TRAIN_H

#include "net.h"


class Train
{
public:
	Train();
	//����ѵ���� 
	Train(const MatrixXd&, const MatrixXd&);
	//ѵ������ 
	bool train(Net&, int);
	//��������ѵ�������� 
	bool calculate(Net&); 
	~Train();
private:
	MatrixXd x_train;
	MatrixXd y_train;
	//����ÿ������ 
	vector<MatrixXd> z;
	//����ÿ������� 
	vector<MatrixXd> e;
};

#endif
