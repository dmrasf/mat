#include "svm.h"
#include <algorithm>

Svm::Svm(double C, double tol, double det) : C(C), tol(tol), det(det) {
} 

void Svm::train(const MatrixXd &x, const MatrixXd &y){
	x_train = x;
	y_train = y;
	a.setZero(1, x.cols());
	b = 0;
	int numChanged = 0;
	int examineAll = 1;
	// 最终的收敛条件是 位于界内的a都满足KKT条件
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
}

int Svm::examineExample(int i2){
	double y2 = y_train(0, i2);
	double a2 = a(i2);
	double E2 = get_E(i2);
	double r2 = E2*y2;
	if((r2 < -tol && a2 < C) || (r2 > tol && a2 > 0)){
		int i = 0;
		for(; i != x_train.cols(); i++)
			if(a(i) > 0 && a(i) < C)
				break;
		if(i != x_train.cols()){
			int i1 = find_second_i(i2);
			if(takeStep(i1, i2))
				return 1;
		}
		
		int i1 = rand();
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
	double E2 = get_E(i2);
	double a1 = a(i1);
	double y1 = y_train(0, i1);
	double E1 = get_E(i1);
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
	if(abs(L == H) < 0.000001)
		return 0;
	double k11 = Gaussian_kernel(x_train.col(i1), x_train.col(i1));
	double k12 = Gaussian_kernel(x_train.col(i1), x_train.col(i2));
	double k22 = Gaussian_kernel(x_train.col(i2), x_train.col(i2));
	double eta = 2*k12 - k11 - k22;
	double a2_new = 0;
	if(eta < 0){
		a2_new = a2 - y2*(get_E(i1) - get_E(i2))/eta;
		if(a2_new < L) 		a2_new = L;
    	else if(a2_new > H) a2_new = H;
	}
	else{
		if(L > H) 			a2_new = L;
		else if(L < H)		a2_new = H;
		else				a2_new = a2;
	}
	if(abs(a2 - a2_new) < 0.001)
		return 0;
	double a1_new = a1 + s * (a2 - a2_new);
	double b1 = b - get_E(i1) - y1*(a1_new - a1)*Gaussian_kernel(x_train.col(i1), x_train.col(i1)) - y2*(a2_new - a2)*Gaussian_kernel(x_train.col(i2), x_train.col(i1));
	double b2 = b - get_E(i2) - y1*(a1_new - a1)*Gaussian_kernel(x_train.col(i1), x_train.col(i2)) - y2*(a2_new - a2)*Gaussian_kernel(x_train.col(i2), x_train.col(i2));
	if(a1_new > 0 && a1_new < C)
		b = b1;
	else if(a2_new > 0 && a2_new < C)
		b = b2;
	else
		b = (b1 + b2) / 2;
	a(i1) = a1_new;
	a(i2) = a2_new;
	return 1;
}

//找到另一个 i 最大 E1 - E2 
int Svm::find_second_i(int i2){
	int i1 = -1;
	double maxE = -1;
	for(int i = 0; i != x_train.cols(); i++){
		double temp = abs(get_E(i2) - get_E(i));
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
	VectorXd temp = (xi.array() - xj.array()).square().colwise().sum();
	temp = (temp/(-2*det*det)).array().exp();
	return temp(0);
}

Svm::~Svm(){
}

