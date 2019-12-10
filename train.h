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
	//����ѵ���� 
	Train(const MatrixXd&, const MatrixXd&);
	//ѵ������ 
	//�ۻ�BP 
	bool train(Net&, int);
	//����ѵ�� 
	bool train_BP(Net&, int);
	bool train(Rbf&, int);
	bool train(Clustering&, int);
	bool train(Svm&);
	//��������ѵ�������� 
	bool calculate(Net&);
	MatrixXd get_d(Net&, int, const MatrixXd&);
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
