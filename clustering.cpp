#include "clustering.h"

Clustering::Clustering(){
}

Clustering::Clustering(const MatrixXd &cla, const VectorXd &lab){
	clas = cla;
	k = lab;
	if(clas.cols() != k.size())
		cout << "类别与初始化向量不一样多" << endl; 
}

Clustering::Clustering(int cla, int input){
	MatrixXd x(input, cla);
	clas = x;
	clas.setRandom();
}

Clustering::~Clustering(){
} 

void Clustering::train(const MatrixXd &x_train, const MatrixXd &y_train, int n, const string &methods){
	if(!methods.compare("k_means"))
		k_means(x_train, n);
	else if(!methods.compare("lvq"))
		lvq(x_train, y_train, n);
	else
		cout << "没有合适的聚类算法" << endl;
}
void Clustering::k_means(const MatrixXd &x_train, int n){
	MatrixXd sum = clas;
	VectorXd cla_num(clas.cols());
	//n:训练次数 
	while(n--){
		sum.setZero();
		cla_num.setZero(); 
		//对每个数据分类 
		for(int i = 0; i != x_train.cols(); i++){
			int min_cla = find_min(x_train.col(i));
			sum.col(min_cla) = sum.col(min_cla).array() + x_train.col(i).array();
			cla_num(min_cla) = cla_num(min_cla) + 1;
		}
		//利用分好类求新的中心向量 
		for(int i = 0; i != clas.cols(); i++){
			if(cla_num(i) != 0)
				clas.col(i) = sum.col(i)/cla_num(i);
		}
	}
}
void Clustering::lvq(const MatrixXd &x_train, const MatrixXd &y_train, int n){
	while(n--){
			
	}
}
//寻找最近的中心向量的函数 
int Clustering::find_min(const VectorXd &x_test){
	MatrixXd len = clas;
	VectorXd u; 
	for(int i = 0; i != len.cols(); i++)
		len.col(i) = len.col(i).array() - x_test.array();
	//求差 然后平方和 
	len = len.array().square();
	u = len.colwise().sum();
	MatrixXd::Index minRow, minCol;
	u.minCoeff(&minRow,&minCol);
	return minRow;
}
//预测函数 
VectorXd Clustering::predict(const MatrixXd &x_test){
	VectorXd y_pre(x_test.cols());
	y_pre.setZero();
	//对要预测的每个数据寻找最近的中心向量 得到的第n类 用k(n)返回预测值 
	for(int i = 0; i != x_test.cols(); i++){
		y_pre(i) = k(find_min(x_test.col(i)));
	}
	return y_pre;
} 

