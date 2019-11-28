#include "net.h"

//ί�й��� 
Net::Net() : Net(1) {}

//��ʼ��������� 
Net::Net(int n) : Input(n) { 
    add_init_x(n);
}

Net& Net::operator = (const Net &net){
	if(this != &net){
		func = net.func;
		layers = net.layers;
		weights= net.weights;
	}
	return *this;
}

Net::Net(const Net &net){
	func = net.func;
	layers = net.layers;
	weights= net.weights;
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
    a.setRandom();
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
MatrixXd Net::d_sigmoid(const MatrixXd &mat){
	return mat.array()*(1 - mat.array()).array();
}

MatrixXd Net::relu(const MatrixXd &input){
	
}
double Net::d_relu(const MatrixXd &mat){
	
}

MatrixXd Net::predict(const MatrixXd &x_test){
	int i = 0;
	MatrixXd temp = x_test;
	for(; i != weights.size(); i++){
		MatrixXd w = weights[i];
		VectorXd b = layers[i+1];	
		MatrixXd e = w*temp;
		for(int j = 0; j != e.cols(); j++)
			e.col(j) = e.col(j) - b;
		MatrixXd z;
		std::string fuc = func[i];
		switch(fuc[0]){
			case 's':
				z = sigmoid(e);
				break;
			case 'r':
				z = relu(e); 
			default:
				z = sigmoid(e);
				break;
		}	
		temp = z;
	}
	return temp;
}

bool Net::calculate(MatrixXd &x, vector<MatrixXd> &in, vector<MatrixXd> &out){
	//xΪ������� �ж���  in Ϊÿ�������  out Ϊÿ������ 
	int i = 0;
	in.clear();
	out.clear();
	out.push_back(x);
	for(; i != weights.size(); i++){
		//��ǰ��Ȩֵ 
		MatrixXd w = weights[i];
		//��ǰ����ֵ 
		VectorXd b = layers[i+1];	
		MatrixXd e = w*out[i];
		for(int j = 0; j != e.cols(); j++)
			e.col(j) = e.col(j) - b;
		MatrixXd z;
		//��ǰ��ʹ�õļ���� 
		std::string fuc = func[i];
		switch(fuc[0]){
			case 's':
				z = sigmoid(e);
				break;
			case 'r':
				z = relu(e);
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
//������� 
bool Net::save_par(const string &path){
	ofstream out(path);
	//�������� 
	for(const auto c : layers)
		out << c.size() << " ";
	out << endl;
	for(int i = 1; i < layers.size(); i++){
		out << layers[i].transpose() << endl;
		out << weights[i-1] << endl;
	}
	out.close();
	return true;
}

bool Net::load_par(const string &path){
	ifstream in(path);
	//��һ�б�Ҫ�Ĳ���  ��һ��Ϊ�ܲ���   ����������Ϊÿ�����Ԫ�ĸ��� 
	vector<int> lays; 
	string pars,par;
	getline(in, pars);
	istringstream line(pars);
	while(getline(line, par, ' ')){
		lays.push_back(stoi(par));
	}
	//�����ԭ�еĲ��� 
	layers.clear();
	weights.clear(); 
	func.clear();
	Input = lays[0];
	Output = lays.back();
	add_init_x(Input);
	//���ݶ�ȡ���Ĳ���һ��һ��Ķ�����
	for(int i = 1; i != lays.size(); i++){
		func.push_back("sigmoid");
		int n_pre = lays[i-1];
		int n_now = lays[i];
		double b[n_now];
		double w[n_now*n_pre];
		//��ȡb 
		getline(in, pars);
		istringstream line(pars);
		int j = 0;
		while(getline(line, par, ' ')){
			if(par.empty())
				continue;
			b[j] = stod(par);
			j++;
		}
		//��Ӳ� 
		VectorXd lay = Map<VectorXd>(b, j);
		layers.push_back(lay);
		j = 0;
		while(getline(in, pars)){
			istringstream line(pars);
			while(getline(line, par, ' ')){
				if(par.empty())
					continue;
				w[j] = stod(par);
				j++;
			}
			if(j == n_now*n_pre)
				break;
		}
		MatrixXd lay_w = Map<MatrixXd, 0, InnerStride<1>>(w, n_pre, n_now);
		weights.push_back(lay_w.transpose());
	} 
	in.close();
	return true;
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
	return layers[lay+1];
}

