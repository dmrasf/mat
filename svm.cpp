#include "svm.h"


Svm::Svm(const MatrixXd &x, const MatrixXd &y){
	
} 


//求第i个数据的a 
double Svm::solve_a(int i){
	VectorXd a_t = a;
	//固定其他值 
	a_t.setOnes();
	a_t(i) = 0;
	double ai = 0;
	//输入数据量  m个 
	int m = x.cols();
	double a_sum = 0;
	for(int i = 0; i != m; i++){
		a_sum +=  
	} 
}





double Svm::Gaussian_kernel(const VectorXd &xi, const VectorxXd &xj){
	VectorXd temp = (xi.array() - xj.array()).square().sum();
	temp = temp.array().exp()/(-2*1);
	return temp(0);
	
}

Svm::~Svm(){
}

