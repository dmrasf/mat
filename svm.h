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
	Svm(const MatrixXd&, const MatrixXd&); 
	//��˹�˺��� 
	double Gaussian_kernel(const VectorXd&, const VectorXd&);
	
	void train(); 
	//��a  SMO
	double solve_a(int); 
	~Svm();
private:
	//��Ϊ0��a  
	VectorXd a;
	MatrixXd y;
	//֧������ 
	MatrixXd x;
	//����  ����   ��Ҫ������õ� 
	int C; 
};

#endif
