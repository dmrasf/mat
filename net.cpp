#include "net.h"

//委托构造 
Net::Net() : Net(1) {}

//初始化输入个数 
Net::Net(int n) : Input(n) { 
    add_init_x(n);
}

//添加层 
bool Net::add_lay(int n, const string &fuc){
	func.push_back(fuc);
	Output = n;
    //前一层的大小 
	int m = (*(l.end() - 1)).size();
    VectorXd v(n);
    v.setRandom();
    l.push_back(v);
    //添加系数 
    MatrixXd a(n, m);
    w.push_back(a);
    return true;
}

int Net::get_NUM_LAY() const {
	return l.size() - 1;
}

int Net::get_NUM_PAR() const {
	int sum = 0;
	for(auto i = l.cbegin() + 1; i != l.cend(); i++)
		sum += (*i).size();
	for(auto i : w)
		sum += i.size();
	return sum;
}

const vector<string>& Net::get_FUNC(){
	return func;
} 

bool Net::add_init_x(int n){
    VectorXd x(n);
    // 
	x.setOnes();
    l.push_back(x);
}

Net::~Net(){}

MatrixXd Net::sigmoid(const MatrixXd &input){
	return (1/((-input).array().exp() + 1)).matrix();
}

MatrixXd Net::relu(const MatrixXd &input){
	
}

double Net::predict(const MatrixXd &x_test, const MatrixXd &y_test){
	//每层的输出 
	MatrixXd temp = x_test;
	for(auto it_l = l.cbegin() + 1, it_w = w.cbegin(); it_w != w.cend(); it_l++, it_w++){
		temp = sigmoid((*it_w)*temp + (*it_l));
	}
	auto y_pre = temp;
	return 0;
}

bool Net::calculate(MatrixXd &x, vector<MatrixXd> &in, vector<MatrixXd> &out){
	//x为输入矩阵 有多组  in 为每层的输入 
	for(int i = 0; i != w.size(); i++){
		auto we = w[i];
		auto b = l[i+1];
		fuc = func[i];
		auto e = we*x + b;
		switch(fuc){
			case "sigmoid":
				auto z = sigmoid(e1);
				break;		
			default:
				auto z = sigmoid(e1);
				break;
		}	
		in.push_back(e);
		out.push_back(z);
	}
	return i == w.size() && i+1 == l.size();
}


