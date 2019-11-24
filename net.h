#ifndef NET_H
#define NET_H

#include <vector>
#include "Eigen/Dense"

using namespace Eigen;
using namespace std;



//保存所有待训练的参数 
class Net{
public:
    Net();
    Net(int);
    bool add_lay(int, const string &fuc = "sigmoid");
    //计算所有参数 
	int get_NUM_PAR() const;
	//层数
	int get_NUM_LAY() const;
	//每层的函数
	const vector<string>& get_FUNC(); 
	//预测
	double predict(const MatrixXd&, const MatrixXd&);
	//计算过程的输入输出 
	bool calculate(MatrixXd&, vector<MatrixXd>&, vector<MatrixXd>&); 
	//函数 
	MatrixXd sigmoid(const MatrixXd&);
	MatrixXd relu(const MatrixXd&);

	~Net();
private:
	//初始化输入 
    bool add_init_x(int);
    //层  包括输入 
    vector<VectorXd> l;
    //每层所用的函数
	vector<string> func; 
	//权值  
    vector<MatrixXd> w;
	int Input;
	int Output; 
};

#endif
