#include "train.h"

#include <iostream>
using namespace std;

Train::Train() : Train(0.3) {}

Train::Train(double rate) : rate(rate) {}

void Train::get_new(const MatrixXd &x, const MatrixXd &y){
	x_train = x;
	y_train = y;
}

Train::Train(const MatrixXd &x, const MatrixXd &y){
	x_train = x;
	y_train = y;
} 
//��������ѵ��������
bool Train::calculate(Net &net){
	net.calculate(x_train, e, z);
}

//ѵ������very important 
bool Train::train(Net &net, int n){
	//n:ѵ��������  net:��ѵ��������  �ۻ�BP 
	while(n > 0){
		n--;
		//��һ��ÿ����������  z:���(x)   e:���� 
		calculate(net);
		//  g = (y_train - z)*z*(1-z)   1 x m
		//  for(every layer) 2 1 0  >0:break
		//		w_e = rate*g*z_next'    [1, m] x [3, m]' = [1, 3]
		//		g = z_next.*(1 - z_next).*(g'*w)'  [3, m] x [[1, m]' x [1, 3]]' = [3, m]
		//		w = w - w_e/m
		//	end
		//
		MatrixXd g = (z.back() - y_train).array()*z.back().array()*(1 - z.back().array()).array();
//		cout << n << " : " << endl; 
		MatrixXd w_e, b_e;
		for(int i = net.get_NUM_LAY(); i > 0; i--){  //2 1 
			//ƫ��			 
			w_e = g.matrix()*z[i-1].transpose();
			b_e = g.matrix();
			//��һ��g 
			MatrixXd temp = g.transpose().matrix()*net.get_w(i-1);
			g = z[i-1].array()*(1 - z[i-1].array()).array()*temp.transpose().array();
			//���²��� 
			MatrixXd w_new = net.get_w(i-1).array() - rate*w_e.array()/x_train.cols();
			VectorXd b_new = net.get_b(i-1).array() + rate*b_e.array().rowwise().mean();
			net.update_w(i-1, w_new);
			net.update_b(i-1, b_new);
		}
	} 
}

bool Train::train(Rbf &rbf, int n){
	MatrixXd c = rbf.get_c();
	//���� c.rows() ��, ��ÿ�����һ��vector 
	vector<vector<VectorXd>> temp(c.rows());
	//���� ��Ҫ���������µ�c��b 
	VectorXd sum(c.rows());
	sum.setZero(); 
	MatrixXd c_new(x_train.rows(), c.rows());
	c_new.setZero();
	//���ಢ����ƽ��ֵ 
	for(int i = 0; i != x_train.cols(); i++){
		//Ѱ����С������� 
		int min_c = rbf.find_min(x_train.col(i));
		c_new.col(min_c) = c_new.col(min_c).array() + x_train.col(i).array();
		//���浽���� 
		temp[min_c].push_back(x_train.col(i));
		//��Ӧ��ļ�����һ 
		sum(min_c) = sum(min_c) + 1;
	}
	VectorXd b_new(c.rows());
	b_new.setZero();
	//����ÿ�����е�ÿ��Ԫ�� 
	for(int i = 0; i != temp.size(); i++){
		//��ȡƽ��ֵ 
		c_new.col(i) = c_new.col(i).array() / sum(i);
		for(const VectorXd &x : temp[i]){
			b_new(i) = x.transpose() * c_new.col(i);
		}
		b_new(i) = b_new(i) / sum(i);
	}
	rbf.update_c(c_new.transpose());
	rbf.update_b(b_new);
	//�������  ����(z[0])�������(z[1])   
	rbf.calculate(x_train, z);
	MatrixXd w_new = rbf.get_w();
	auto e = z[1].array() - y_train.array();
	w_new = w_new.array() - rate*(e.matrix()*z[0].transpose()).array();
	rbf.update_w(w_new);
}

void Train::show_cal(){
	cout << "z0 = " << endl;
	cout << z[0] << endl;
	for(int i = 0; i != e.size(); i++){
		cout << "e" << i << " = " << endl;
		cout << e[i] << endl;
		cout << "z" << i+1 << " = " << endl;
		cout << z[i+1] << endl;
	}
}

Train::~Train(){
}

