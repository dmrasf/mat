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
	//n:训练集个数  net:待训练的网络  累积BP 
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
		
		MatrixXd d_func = get_d(net, net.get_NUM_LAY(), z.back());
		MatrixXd g = (z.back() - y_train).array()*d_func.array();
//		MatrixXd g = (z.back() - y_train).array();//*z.back().array()*(1 - z.back().array()).array();
//		cout << n << " : " << endl; 
		MatrixXd w_e, b_e;
		for(int i = net.get_NUM_LAY(); i > 0; i--){  //2 1 
			//偏导			 
			w_e = g.matrix()*z[i-1].transpose();
			b_e = g.matrix();
			//下一个g 
			d_func = get_d(net, i-1, z[i-1]);
			MatrixXd temp = g.transpose().matrix()*net.get_w(i-1);
//			g = z[i-1].array()*(1 - z[i-1].array()).array()*temp.transpose().array();
			g = d_func.array()*temp.transpose().array();
			//更新参数 
			MatrixXd w_new = net.get_w(i-1).array() - rate*w_e.array()/x_train.cols();
			VectorXd b_new = net.get_b(i-1).array() + rate*b_e.array().rowwise().mean();
			net.update_w(i-1, w_new);
			net.update_b(i-1, b_new);
		}
	} 
}

MatrixXd Train::get_d(Net &net, int lay, const MatrixXd &z){
	if(lay == 0){
		return z;
	}
	string func = net.get_FUNC(lay-1);
	if(!func.compare("sigmoid"))
		return net.d_sigmoid(z);
	else if(!func.compare("relu"))
		return net.d_relu(z);
	else if(!func.compare("linear"))
		return net.d_linear(z);
	else
		return net.d_sigmoid(z);
}

bool Train::train(Rbf &rbf, int n){
	MatrixXd c = rbf.get_c();
	VectorXd sum(c.rows());
	sum.setZero(); 
	MatrixXd c_new(c.rows(), x_train.rows()), len(c.rows(), x_train.cols());
	c_new.setZero();
	//归类并计算平均值  ？好像不应该随机初始化聚类中心  
	for(int i = 0; i != x_train.cols(); i++){
		//寻找最小距离的类 
		int min_c = rbf.find_min(x_train.col(i));
		c_new.row(min_c) = c_new.row(min_c).array() + x_train.col(i).transpose().array();
		//对应类的计数加一 
		sum(min_c) = sum(min_c) + 1;
		len.col(i) = rbf.len(x_train.col(i));
	}
	//遍历每个类中的每个元素  求取平均值 
	for(int i = 0; i != sum.size(); i++){
//		c_new.row(i) = (c_new.row(i).array() + c.row(i).array()) / (sum(i)+1);
		c_new.row(i) = c_new.row(i).array() / sum(i);
	}
	//计算输出  隐层(z[0])和输出层(z[1])   
	rbf.calculate(x_train, z);
	//用BP算w和b
	MatrixXd u = z[0], y_output = z[1];
	MatrixXd E = y_output.array() - y_train.array();
	MatrixXd w_new = rbf.get_w().array() - rate*(E*u.transpose()).array()/x_train.cols();
	VectorXd b_new = rbf.get_b().array() - rate*((rbf.get_w().transpose()*E).array()*u.array()*len.array()).rowwise().mean()/rbf.get_b().array().cube();
	for(int i = 0; i != sum.size(); i++){
		if(sum(i) != 0){
			c.row(i) = c_new.row(i);
		}
	}	
	
//	MatrixXd temp_b = rbf.get_b(), temp_c  = rbf.get_c();
//	temp_b.setZero();
//	temp_c.setZero();
//	for(int i = 0; i != x_train.cols(); i++){
//		temp_b = rate*((rbf.get_w().transpose()*E).array()*u.array()).rowwise().mean()/rbf.get_b().array();
//		MatrixXd x_c = rbf.get_c();
//		for(int i = 0; i != x_train.rows(); i++){
//			x_c.row(i) = x_train.col(i).transpose().array() - x_c.row(i).array();
//			temp_c.row(i) = temp_c.row(i).array() + x_c.row(i).array()*temp_b.row(i).array();	
//		}
//	}
//	c = temp_c/x_train.cols();

	rbf.update_c(c);
	rbf.update_b(b_new);
	rbf.update_w(w_new);
}

bool Train::train(Clustering &clus, int n){
	clus.train(x_train, y_train, n, "lvq");
}

bool Train::train(Svm &svm){
	svm.train(x_train, y_train);
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

