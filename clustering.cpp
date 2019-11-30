#include "clustering.h"

Clustering::Clustering(){
}

Clustering::Clustering(const MatrixXd &cla, const VectorXd &lab){
	clas = cla;
	k = lab;
	if(clas.cols() != k.size())
		cout << "������ʼ��������һ����" << endl; 
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
		cout << "û�к��ʵľ����㷨" << endl;
}
void Clustering::k_means(const MatrixXd &x_train, int n){
	MatrixXd sum = clas;
	VectorXd cla_num(clas.cols());
	//n:ѵ������ 
	while(n--){
		sum.setZero();
		cla_num.setZero(); 
		//��ÿ�����ݷ��� 
		for(int i = 0; i != x_train.cols(); i++){
			int min_cla = find_min(x_train.col(i));
			sum.col(min_cla) = sum.col(min_cla).array() + x_train.col(i).array();
			cla_num(min_cla) = cla_num(min_cla) + 1;
		}
		//���÷ֺ������µ��������� 
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
//Ѱ����������������ĺ��� 
int Clustering::find_min(const VectorXd &x_test){
	MatrixXd len = clas;
	VectorXd u; 
	for(int i = 0; i != len.cols(); i++)
		len.col(i) = len.col(i).array() - x_test.array();
	//��� Ȼ��ƽ���� 
	len = len.array().square();
	u = len.colwise().sum();
	MatrixXd::Index minRow, minCol;
	u.minCoeff(&minRow,&minCol);
	return minRow;
}
//Ԥ�⺯�� 
VectorXd Clustering::predict(const MatrixXd &x_test){
	VectorXd y_pre(x_test.cols());
	y_pre.setZero();
	//��ҪԤ���ÿ������Ѱ��������������� �õ��ĵ�n�� ��k(n)����Ԥ��ֵ 
	for(int i = 0; i != x_test.cols(); i++){
		y_pre(i) = k(find_min(x_test.col(i)));
	}
	return y_pre;
} 

