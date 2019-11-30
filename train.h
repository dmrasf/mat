#ifndef TRAIN_H
#define TRAIN_H
#include "net.h"
#include "rbf.h"
#include "clustering.h"

class Train
{
public:
	Train();
	Train(double); 
	void get_new(const MatrixXd&, const MatrixXd&);
	//����ѵ���� 
	Train(const MatrixXd&, const MatrixXd&);
	//ѵ������ 
	bool train(Net&, int);
	bool train(Rbf&, int);
	bool train(Clustering&, int);
	//��������ѵ�������� 
	bool calculate(Net&);
	//test
	void show_cal();
	~Train();
private:
	MatrixXd x_train;
	MatrixXd y_train;
	//����ÿ������ 
	vector<MatrixXd> z;
	//����ÿ������� 
	vector<MatrixXd> e;
	double rate = 0.3;
};

#endif
