#include "clustering.h"

//Clustering::Clustering(){
//}
//用已知类别的数据初始化 
Clustering::Clustering(const MatrixXd &cla, const VectorXd &lab){
	clas = cla;
	k = lab;
	if(clas.cols() != k.size())
		cerr << "类别与初始化向量不一样多" << endl; 
}
//cla:要分的类别   input:输入维度 
Clustering::Clustering(int cla, int input){
	MatrixXd x(input, cla);
	clas = x;
	clas.setRandom();
	VectorXd y(cla);
	for(int i = 0; i != cla; i++)
		y(i) = i;
	k = y;
}

Clustering::~Clustering(){
} 

void Clustering::train(const MatrixXd &x_train, const MatrixXd &y_train, int n, const string &methods){
	if(!methods.compare("k_means"))
		k_means(x_train, y_train, n);
	else if(!methods.compare("lvq"))
		lvq(x_train, y_train, n);
	else
		cerr << "没聚类算法" << endl;
}
void Clustering::k_means(const MatrixXd &x_train, const MatrixXd &y_train, int n){
	MatrixXd sum = clas;
	VectorXd cla_num(clas.cols());
	//s 选择每一类中最多的标记作为新的类别   row:类别  col:遍历数据后在各类别出现的次数 
	MatrixXd s(clas.cols(), clas.cols());
	s.setZero();
	//n:训练次数 
	while(n--){
		sum.setZero();
		cla_num.setZero(); 
		//对每个数据分类 
		for(int i = 0; i != x_train.cols(); i++){
			int min_cla = find_min(x_train.col(i));
			sum.col(min_cla) = sum.col(min_cla).array() + x_train.col(i).array();
			cla_num(min_cla) = cla_num(min_cla) + 1;
			s(min_cla, y_train(0, i)) = s(min_cla, y_train(0, i)) + 1;
		}
		//利用分好类求新的中心向量 
		for(int i = 0; i != clas.cols(); i++){
			if(cla_num(i) != 0)
				clas.col(i) = sum.col(i)/cla_num(i);
			//更新类别	 ?*********?
			//对现有的每一类 求最大值（出现的次数最多）所在在的列 即为新的类别 这样就不用自己去分了 
			MatrixXd::Index maxRow, maxCol;
			s.row(i).maxCoeff(&maxRow,&maxCol);
			k(i) = maxCol;
		}
//		cout << "n = " << n << endl;
	}
}
void Clustering::lvq(const MatrixXd &x_train, const MatrixXd &y_train, int n){
	while(n--){
		for(int i = 0; i != x_train.cols(); i++){
			auto x = x_train.col(i);
			double y = y_train(0, i);
			int min_dis = find_min(x);
			cout << "y = " << y << "  min = " << min_dis << "  k = " << k(min_dis) << endl;
			if(k(min_dis) == y)
				clas.col(min_dis) = clas.col(min_dis).array() + rate*(x.array()-clas.col(min_dis).array());
			else
				clas.col(min_dis) = clas.col(min_dis).array() - rate*(x.array()-clas.col(min_dis).array());
		}
		cout << n << endl;
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

