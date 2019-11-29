#include "rbf.h"

Rbf::Rbf() : Rbf(1, 2, 1){
}

Rbf::Rbf(int x_n, int lay_n, int y_n){
	c.resize(lay_n, x_n);
	c.setRandom();
	b.resize(lay_n);
	b.setRandom();
	b = b.array().abs();
	w.resize(y_n, lay_n);
	w.setRandom();
}

Rbf& Rbf::operator = (const Rbf &rbf){
	if(this != &rbf){
		c = rbf.c;
		b = rbf.b;
		w = rbf.w;
	}
	return *this;
}
Rbf::Rbf(const Rbf &rbf){
	c = rbf.c;
	b = rbf.b;
	w = rbf.w;
}
void Rbf::resize(int x_n, int lay_n, int y_n){
	c.resize(lay_n, x_n);
	c.setRandom();
	b.resize(lay_n);
	b.setRandom();
	b = b.array().abs();
	w.resize(y_n, lay_n);
	w.setRandom();
}

Rbf::~Rbf(){
}

const MatrixXd& Rbf::get_c(){
	return c;
}
const MatrixXd& Rbf::get_w(){
	return w;
}
void Rbf::update_c(const MatrixXd &c_new){
	c = c_new;
}
void Rbf::update_b(const VectorXd &b_new){
	b = b_new;
}
void Rbf::update_w(const MatrixXd &w_new){
	w = w_new;
}

//高斯核函数 
VectorXd Rbf::Gaussian_kernel(const VectorXd &x){
	MatrixXd temp = c;
	VectorXd u;
	for(int i = 0; i != temp.rows(); i++){
		temp.row(i) = temp.row(i).array() - x.transpose().array();
	}
	temp = temp.array().square();
	u = temp.rowwise().sum().array() / (b.array()*(-2));
	u = u.array().exp();
	return u;
} 

bool Rbf::calculate(MatrixXd &x, vector<MatrixXd> &out){
	//x为输入矩阵 有多组 out u d
	out.clear();
	MatrixXd u(b.size(), x.cols());
	for(int i = 0; i != x.cols(); i++){
		u.col(i) = Gaussian_kernel(x.col(i));
	}
	out.push_back(u);
	auto b = w*u;
	out.push_back(b);
	return true;
}

int Rbf::find_min(const VectorXd &x){
	MatrixXd temp = c*x;
	MatrixXd::Index minRow, minCol;
	temp.minCoeff(&minRow,&minCol);
	return minRow;
}






