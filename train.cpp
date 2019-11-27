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
//计算用于训练的数据
bool Train::calculate(Net &net){
	net.calculate(x_train, e, z);
}

//训练函数very important 
bool Train::train(Net &net, int n){
	//n:训练集个数  net:待训练的网络
	while(n > 0){
		n--;
		//求一下每层的输入输出  z:输出(x)   e:输入 
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
			//偏导			 
			w_e = g.matrix()*z[i-1].transpose();
			b_e = g.matrix();
			
			//下一个g 
			MatrixXd temp = g.transpose().matrix()*net.get_w(i-1);
			g = z[i-1].array()*(1 - z[i-1].array()).array()*temp.transpose().array();
			
			//更新参数 
			MatrixXd w_new = net.get_w(i-1).array() - rate*w_e.array()/x_train.cols();
			VectorXd b_new = net.get_b(i-1).array() + rate*b_e.array().rowwise().mean();
			net.update_w(i-1, w_new);
			net.update_b(i-1, b_new);
		}
	} 
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

