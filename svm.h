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
	Svm(double, double, double);
 	void train(const MatrixXd&, const MatrixXd&);
	//��˹�˺��� 
	double Gaussian_kernel(const VectorXd&, const VectorXd&);
	VectorXd get_a();
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
	VectorXd E;
	double C;
	double tol;
	double det;
	double b = 0;
};

#endif
