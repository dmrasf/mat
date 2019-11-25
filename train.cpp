#include "train.h"

#include <iostream>
using namespace std;

Train::Train(){
	
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
	//n:迭代次数  net:待训练的网络
	while(n > 0){
		n--;
		//求一下每层的输入输出  z:输出(x)   e:输入 
		calculate(net);
		//当前 wi = wi - a*temp/z_i*w_(i+1)*(1-z_i)*z_i*z_(i-1)
		//BP 
//		auto lay_pre = (z - z_)*z(i-1) 
//		int w_pre = 1;
//		for(int i = net.get_NUM_LAY() - 1; i >= 0; i--){
//			int i_z = i+1;
//			int i_e = i;
//			auto m = (1 - z[i_z]).array()*z[i_z].array();
//			lay_pre/z[i_z]*w_pre*m;
//			w_pre = 
//			net.add_lay();
//			
//		}
		
		//g_pre = (y_tr-y).*y.*(1-y)
		//for(1)
		//rate*g_pre*z(i-1)
		//g_pre = y_n*(1-y_n)*w*g_pre 
		//end
		//  g = (y_train - z)*z*(1-z)   1 x m
		//  for(every layer) 2 1 0  >0:break
		//		w_e = rate*g*z_next'    [1, m] x [3, m]' = [1, 3]
		//		g = z_next.*(1 - z_next).*(g'*w)'  [3, m] x [[1, m]' x [1, 3]]' = [3, m]
		//		w = w - w_e/m
		//	end
		//
		auto g = (y_train - z.back()).array()*z.back().array()*(1 - z.back().array()).array();
		MatrixXd w_e;
		for(int i = net.get_NUM_LAY(); i > 0; i--){  //2 1 
			//偏导 
			w_e = rate*g.matrix()*z[i-1].transpose();
			auto temp = g.transpose().matrix()*net.get_w(i);
			g = z[i-1].array()*(1-z[i-1].array()).array()*temp.transpose().array();
			
			
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

