#include "svm.h"


Svm::Svm(const MatrixXd &x, const MatrixXd &y){
	
} 


//���i�����ݵ�a 
double Svm::solve_a(int i){
	VectorXd a_t = a;
	//�̶�����ֵ 
	a_t.setOnes();
	a_t(i) = 0;
	double ai = 0;
	//����������  m�� 
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

