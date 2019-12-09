#include "svm.h"
#include <algorithm>
#include <cstdlib>
#include <ctime> 

Svm::Svm(){
}

Svm::Svm(double C, double tol, double det) : C(C), tol(tol), det(det) {
} 

const VectorXd& Svm::get_a(){
	return a;
}
const VectorXi& Svm::get_pos(){
	return pos;
}
double Svm::get_b(){
	return b;
}

int Svm::get_train_num(){
	return x_train.cols()/2;
}

void Svm::load(VectorXd &a, VectorXi &pos, double b, const MatrixXd &x, const MatrixXd &y){
	this->a = a;
	this->b = b;
	this->pos = pos;
	this->x_train = x;
	this->y_train = y;
}

VectorXd Svm::predict(const MatrixXd &x_test){
	VectorXd y_pre(x_test.cols());
	y_pre.setZero();
	for(int i = 0; i != x_test.cols(); i++){
		double temp = 0;
		for(int j = 0; j != x_train.cols(); j++)
			temp = temp + a(j)*y_train(j)*Gaussian_kernel(x_train.col(j), x_test.col(i));
		y_pre(i) = temp;
	}
	return y_pre.array() + b;
}

void Svm::train(const MatrixXd &x, const MatrixXd &y){
	x_train = x;
	y_train = y;
	a.setZero(x.cols());
	b = 0;
	E = a;
	for(int i = 0; i != x_train.cols(); i++){
		E(i) = get_E(i);
	}
	int numChanged = 0;
	int examineAll = 1;
	// 最终的收敛条件是 位于界内的a都满足KKT条件
	// 支持向量机导论 
	while(numChanged > 0 || examineAll){	
		numChanged = 0;
		if(examineAll){
			for(int i = 0; i != x.cols(); i++)
				numChanged = examineExample(i) + numChanged;
		}
		else{
			for(int i = 0; i != x.cols(); i++)
				if(a(i) != 0 && a(i) != C)
					numChanged = examineExample(i) + numChanged;
		}
		if(examineAll == 1)
			examineAll = 0;
		else if(numChanged == 0)
			examineAll = 1;
	}
	delete_a();
}

int Svm::examineExample(int i2){
	double y2 = y_train(0, i2);
	double a2 = a(i2);
	double E2 = E(i2);
	double r2 = E2*y2;
	//不满足KKT条件 
	if((r2 < -tol && a2 < C) || (r2 > tol && a2 > 0)){
		int i = 0;
		//选择边界内的点先更新 
		for(; i != x_train.cols(); i++)
			if(a(i) > 0 && a(i) < C)
				break;
		if(i != x_train.cols()){
			int i1 = find_second_i(i2);
			if(takeStep(i1, i2))
				return 1;
		}
		int i1 = rand() % x_train.cols();
		if(takeStep(i1, i2))
			return 1;
	}
	return 0;
}

//主要的算法 SMO 
int Svm::takeStep(int i1, int i2){
	if(i1 == i2)
		return 0;
	double a2 = a(i2);
	double y2 = y_train(0, i2);
	double E2 = E(i2);
	double a1 = a(i1);
	double y1 = y_train(0, i1);
	double E1 = E(i1);
	double s = y1*y2; 
	// 更新范围
	double L = 0, H = 0;
	if(y1 == y2){
		L = max(0.0, a1 + a2 - C);
		H = min(C, a1 + a2);
	}
	else{
		L = max(0.0, a2 - a1);
		H = min(C, C - a1 + a2);
	}
	if(L == H)
		return 0;
	double k11 = Gaussian_kernel(x_train.col(i1), x_train.col(i1));
	double k12 = Gaussian_kernel(x_train.col(i1), x_train.col(i2));
	double k22 = Gaussian_kernel(x_train.col(i2), x_train.col(i2));
	double eta = 2*k12 - k11 - k22; 
	double a2_new = 0;
	//eta为f的二阶导数 若eta >= 0 则f的最大值在边界上 
	if(eta < 0){
		a2_new = a2 - y2*(E1 - E2)/eta;
		if(a2_new < L) 		a2_new = L;
    	else if(a2_new > H) a2_new = H;
	}
	else{
		if(L > H) 			a2_new = L;
		else if(L < H)		a2_new = H;
		else				a2_new = a2;
	}
	//更新值很小 
	if(abs(a2 - a2_new) < 0.001)
		return 0;
	double a1_new = a1 + s * (a2 - a2_new);
	double b1 = b - E1 - y1*(a1_new - a1)*Gaussian_kernel(x_train.col(i1), x_train.col(i1)) - y2*(a2_new - a2)*Gaussian_kernel(x_train.col(i2), x_train.col(i1));
	double b2 = b - E2 - y1*(a1_new - a1)*Gaussian_kernel(x_train.col(i1), x_train.col(i2)) - y2*(a2_new - a2)*Gaussian_kernel(x_train.col(i2), x_train.col(i2));
	if(a1_new > 0 && a1_new < C)
		b = b1;
	else if(a2_new > 0 && a2_new < C)
		b = b2;
	else
		b = (b1 + b2) / 2;
	a(i1) = a1_new;
	a(i2) = a2_new;
	for(int i = 0; i != x_train.cols(); i++){
		E(i) = get_E(i);
	}
	cout << "更新E" << endl; 
	return 1;
}

//找到另一个 i 最大 E1 - E2 
int Svm::find_second_i(int i2){
	int i1 = -1;
	double maxE = -1;
	double E2 = get_E(i2);
	for(int i = 0; i != x_train.cols(); i++){
		double temp = abs(E2 - E(i));
		if(maxE < temp){
			maxE = temp;
			i1 = i;
		}
	}
	return i1; 
} 
//求 E(i) = f(i) - y(i) 
double Svm::get_E(int i2){
	double temp = 0;
	for(int i = 0; i != x_train.cols(); i++)
		temp = temp + a(i)*y_train(0, i)*Gaussian_kernel(x_train.col(i), x_train.col(i2));
	return temp + b - y_train(0, i2);
}
//高斯核函数 
double Svm::Gaussian_kernel(const VectorXd &xi, const VectorXd &xj){
	VectorXd temp = (xi.array() - xj.array()).transpose().matrix()*(xi.array() - xj.array()).matrix();
	temp = (temp/(-2*det*det)).array().exp();
	return temp(0);
}

void Svm::delete_a(){
	int sum = (a.array() != 0).count();
	VectorXd a_new(sum);
	pos.resize(sum);
	int j = 0;
	for(int i = 0; i != a.size(); i++){
		if(a(i) != 0){
			pos(j) = i;
			a_new(j) = a(i);
			j++;
		}
	}
	a = a_new;
}

Svm::~Svm(){
}

