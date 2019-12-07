#ifndef NET_H
#define NET_H
#include <vector>
#include "Eigen/Dense"
#include <string>
#include <iostream>
#include <fstream> 
using namespace Eigen;
using namespace std;
//保存所有待训练的参数 
class Net{
public:
    Net();
    Net(int);
    Net& operator = (const Net&);
    Net(const Net&);
    virtual bool add_lay(int, const string &fuc = "sigmoid");
    //计算所有参数  
	int get_NUM_PAR() const;
	//层数
	int get_NUM_LAY() const;
	//每层的函数
	const string& get_FUNC(int); 
	//预测
	MatrixXd predict(const MatrixXd&);
	//计算过程的输入输出 
	bool calculate(MatrixXd&, vector<MatrixXd>&, vector<MatrixXd>&); 
	//更新参数 ****** 
	void update_w(int, MatrixXd&);
	void update_b(int, VectorXd&);
	//保存及拿取训练后的参数
	bool save_par(const string&); 
	bool load_par(const string&);
	//获取参数 更新参数用 
	const MatrixXd& get_w(int);
	const VectorXd& get_b(int); 
	//不同的激活函数 及导 
	MatrixXd sigmoid(const MatrixXd&);
	MatrixXd d_sigmoid(const MatrixXd&); 
	MatrixXd relu(const MatrixXd&);
	MatrixXd d_relu(const MatrixXd&);
	MatrixXd linear(const MatrixXd&);
	MatrixXd d_linear(const MatrixXd&);
	virtual ~Net();
protected://派生类可访问 
	//初始化输入 
    bool add_init_x(int);
    //层  包括输入 
    vector<VectorXd> layers;
    //每层所用的函数
	vector<string> func; 
	//权值  
    vector<MatrixXd> weights;
	int Input = -1;
	int Output = -1; 
};

#endif
