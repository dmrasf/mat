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
	//高斯核函数 
	double Gaussian_kernel(const VectorXd&, const VectorXd&);
	
	void train(); 
	//求a  SMO
	double solve_a(int); 
	~Svm();
private:
	//不为0的a  
	VectorXd a;
	MatrixXd y;
	//支持向量 
	MatrixXd x;
	//常数  软间隔   需要尝试最好的 
	int C; 
};

#endif
