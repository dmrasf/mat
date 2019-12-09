#ifndef SVM_H
#define SVM_H
#include <vector>
#include "Eigen/Dense"
#include <string>
#include <iostream>
#include <fstream> 
using namespace Eigen;
using namespace std;

class Svm
{
public:
	//��ʼ����Ҫ ��һЩ������ֵ 
	Svm(); 
	Svm(double, double, double);
 	void train(const MatrixXd&, const MatrixXd&);
	//��˹�˺��� 
	double Gaussian_kernel(const VectorXd&, const VectorXd&);
	const VectorXd& get_a();
	const VectorXi& get_pos();
	double get_b();
	int get_train_num();
	//��ȡֵ 
	void load(VectorXd&, VectorXi&, double, const MatrixXd&, const MatrixXd&);
	VectorXd predict(const MatrixXd&);
	~Svm();
private:
	int examineExample(int);
	double get_E(int);
	//ֻ����֧������ 
	void delete_a();
	int find_second_i(int);
	int takeStep(int, int);
	VectorXd a;
	MatrixXd y_train; 
	MatrixXd x_train;
	//a!=0��λ�� 
	VectorXi pos; 
	VectorXd E;
	double C = 30;
	double tol = 0.09;
	double det = 5;
	double b = 0;
};

#endif
