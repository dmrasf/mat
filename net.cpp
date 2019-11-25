#include "net.h"

//ί�й��� 
Net::Net() : Net(1) {}

//��ʼ��������� 
Net::Net(int n) : Input(n) { 
    add_init_x(n);
}

//��Ӳ� 
bool Net::add_lay(int n, const string &fuc){
	func.push_back(fuc);
	Output = n;
    //ǰһ��Ĵ�С 
	int m = (*(layers.end() - 1)).size();
    VectorXd v(n);
    v.setRandom();
    layers.push_back(v);
    //���ϵ�� 
    MatrixXd a(n, m);
    weights.push_back(a);
    return true;
}

int Net::get_NUM_LAY() const {
	return layers.size() - 1;
}

int Net::get_NUM_PAR() const {
	int sum = 0;
	for(auto i = layers.cbegin() + 1; i != layers.cend(); i++)
		sum += (*i).size();
	for(auto i : weights)
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
    layers.push_back(x);
}

Net::~Net(){}

MatrixXd Net::sigmoid(const MatrixXd &input){
	return (1/((-input).array().exp() + 1)).matrix();
}

MatrixXd Net::relu(const MatrixXd &input){
	
}

void Net::predict(const MatrixXd &x_test, const MatrixXd &y_test){
	int i = 0;
	MatrixXd temp = x_test;
	for(; i != weights.size(); i++){
		MatrixXd w = weights[i];
		VectorXd b = layers[i+1];	
		MatrixXd e = w*temp;
		for(int j = 0; j != e.cols(); j++)
			e.col(j) = e.col(j) + b;
		MatrixXd z;
		std::string fuc = func[i];
		switch(fuc[0]){
			case 's':
				z = sigmoid(e);
				break;		
			default:
				z = sigmoid(e);
				break;
		}	
		temp = z;
	}
	cout << "y_predict = " << temp << endl;
}

bool Net::calculate(MatrixXd &x, vector<MatrixXd> &in, vector<MatrixXd> &out){
	//xΪ������� �ж���  in Ϊÿ�������  out Ϊÿ������ 
	int i = 0;
	out.push_back(x);
	for(; i != weights.size(); i++){
		//��ǰ��Ȩֵ 
		MatrixXd w = weights[i];
		//��ǰ����ֵ 
		VectorXd b = layers[i+1];	
		MatrixXd e = w*out[i];
		for(int j = 0; j != e.cols(); j++)
			e.col(j) = e.col(j) + b;
		MatrixXd z;
		//��ǰ��ʹ�õļ���� 
		std::string fuc = func[i];
		switch(fuc[0]){
			case 's':
				z = sigmoid(e);
				break;		
			default:
				z = sigmoid(e);
				break;
		}	
		in.push_back(e);
		out.push_back(z);
	}
	return i == weights.size() && i+1 == layers.size();
}

void Net::update_b(int i, VectorXd &new_b){
	layers[i+1] = new_b;
} 

void Net::update_w(int i, MatrixXd &new_w){ 
	weights[i] = new_w;
}

const MatrixXd& Net::get_w(int lay){
	return weights[lay];
}

const VectorXd& Net::get_b(int lay){
	return layers[lay];
}
















