#ifndef RBF_H
#define RBF_H
#include <vector>
#include "Eigen/Dense"
#include <string>
#include <iostream>
#include <fstream> 
using namespace Eigen;
using namespace std;

class Rbf
{
public:
	Rbf();
	//输入层个数  隐层核个数  输出个数 
	Rbf(int, int, int);
    Rbf& operator = (const Rbf&);
    Rbf(const Rbf&);
    void resize(int, int, int);
    
    //找到最近的类
	int find_min(const VectorXd&); 
    
    //参数更新 
    const MatrixXd& get_c();
	const MatrixXd& get_w();
	void update_c(const MatrixXd&);
	void update_b(const VectorXd&);
	void update_w(const MatrixXd&);
    bool calculate(MatrixXd&, vector<MatrixXd>&);
    //高斯核函数 
	VectorXd Gaussian_kernel(const VectorXd&);
	~Rbf();
private:
	//输入到隐层的中心 
	MatrixXd c;
	//高斯核函数的权  >0 
	VectorXd b;
	//隐层到输出的权值 
	MatrixXd w;
};

#endif
