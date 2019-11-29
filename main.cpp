#include <iostream>
#include "Eigen/Dense"
#include "net.h"
#include "rbf.h"
#include "train.h"
#include <fstream> 

using namespace Eigen;
using namespace std;

//预测函数    需要预测的个数 
void predict(Net &net, int n = 10);
//训练函数   训练参数   训练函数    训练次数   一次训练的个数 
void train(Net &net, Train &tra, int n = 1, int m = 5000);

int main()
{
	Rbf r(3,4,2);
	MatrixXd x(3, 2);
	x << -1,0,
		 3,2,
		 4,5;
	vector<MatrixXd> z;
	r.calculate(x, z); 
	for(auto c : z)
		cout << c << endl;
	VectorXd m(3);
	m << 1,2,3;
	cout << r.find_min(m) << endl;
		
//	Net net(784);
//	net.add_lay(50);
//	net.add_lay(10);e
//	cout << "需训练的参数 : " << net.get_NUM_PAR() << "个" << endl;
//	cout << "共有 : " << net.get_NUM_LAY() << "层" << endl;
//	Train tra;
//	//训练前预测 
//	predict(net, 10);
//	//训练前5000个 1次 
//	train(net, tra, 1, 60000);
//	net.save_par( "par.csv");
	
//	Net n;
//	n.load_par("par.csv");
//	predict(n, 1000);

  	return 0; 
}

//预测函数 
void predict(Net &net, int n){

	ifstream r_test_y("test_y.csv");
	int test_y_[10000];
	int i = 0;
	string word;
	while(getline(r_test_y, word, ',')){		
		test_y_[i] = stoi(word);
		i++;
	}
	r_test_y.close();
	double sum = 0.0; 
	ifstream r1("test_x.csv");
	string line;
	double test_x_[784];
	int j = 0;
	while(getline(r1, line)){
		istringstream record(line);
		string word;
		int i = 0;
		while(getline(record, word, ',')){		
			test_x_[i] = stoi(word);
			i++;
		}
		cout << j << endl;
		MatrixXd test_x = Map<Matrix<double, 1, 784>>(test_x_);
		test_x = test_x/255;
		auto y_pre = net.predict(test_x.transpose());
//		cout << Map<Matrix<double, 28, 28, RowMajor>>(test_x_) << endl << endl;
//		cout << "实际值 = " << test_y_[j] << endl;
//		cout << y_pre.transpose() << endl;
		MatrixXd::Index maxRow, maxCol;
		y_pre.maxCoeff(&maxRow,&maxCol);
//		cout << "预测值 = "  << maxRow << endl << endl;
		sum = sum + (test_y_[j] == maxRow);
		j++;
		if(j == n)
			break;
	}
	r1.close();
	cout << "准确率 = " << sum/n << endl;
}
//训练函数 
void train(Net &net, Train &tra, int n, int m){
	
	//读取训练标记数据 
	ifstream r_train_y("train_y.csv");
	int train_y_[m];
	int i = 0;
	string word;
	while(getline(r_train_y, word, ',')){		
		train_y_[i] = stoi(word);
		i++;
		if(i == m)
			break;
	}
	r_train_y.close();
	//定义训练数据 
	MatrixXd train_y(1, 10);
	train_y.setZero();
	MatrixXd train_x(1, 784);
	train_y.setZero();
	//训练次数 
	while(n--){
		ifstream fr("train_x.csv");
		string line;
		double train_x_[784];
		int j = 0;
		while(getline(fr, line)){
			//处理为 10维向量 
			train_y.setZero();
			train_y(0, train_y_[j]) = 1;
			istringstream record(line);
			string word;
			int i = 0;
			while(getline(record, word, ',')){		
				train_x_[i] = stoi(word);
				i++;
			}
			j++;
			train_x = Map<Matrix<double, 1, 784>>(train_x_);
			train_x = train_x/255;
			tra.get_new(train_x.transpose(),train_y.transpose());
			tra.train(net, 1); 
			if(j == m)
				break;
		}
		fr.close();
		cout << "第" << n << "次" <<endl; 
	}
}

//	Net net(8);
//	net.add_lay(3);
//	net.add_lay(3);
//	net.add_lay(1);
//	
//	cout << "需训练的参数 : " << net.get_NUM_PAR() << "个" << endl;
//	cout << "共有 : " << net.get_NUM_LAY() << "层" << endl;
//	
//	MatrixXd x(8,17), y(1,17);
//
//	x << 2,3,3,2,1,2,3,3,3,2,1,1,2,1,3,1,2,
//		 2,2,2,2,2,1,1,1,1,3,3,2,1,1,1,2,2,
//		 2,3,2,3,2,2,2,2,3,1,1,2,2,3,2,2,3,
//		 1,1,1,1,1,1,2,1,2,1,3,3,2,2,1,3,2,
//		 3,3,3,3,3,2,2,2,2,1,1,1,3,3,2,1,2,
//		 1,1,1,1,1,2,2,1,1,2,1,2,1,1,2,1,1,
//		 0.697,0.744,0.634,0.608,0.556,0.403,0.481,0.437,0.666,0.243,0.245,0.343,0.639,0.657,0.360,0.593,0.719,
//		 0.460,0.376,0.264,0.318,0.215,0.237,0.149,0.211,0.091,0.267,0.057,0.099,0.161,0.198,0.370,0.042,0.103;
//    y << 1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0;
//		
//	Train tra(x, y);
//
//	cout << endl << "训练前" << endl;
//	auto y_pre = net.predict(x, y);
//	cout << y_pre << endl;
//	double sum = 0.0;
//	for(int i = 0; i != y_pre.cols(); i++){
//		if(y(0, i) == 1 && y_pre(0, i) > 0.5 || y(0, i) == 0 && y_pre(0, i) < 0.5)
//			sum++;
//	}
//	cout << sum/y.cols() << endl;
//	
//	Net ne(net);
//	
//	for(int i = 1000; i <= 10000; i += 1000){
//		net = ne;
//		tra.train(net, i);
//		auto y_pre = net.predict(x, y);
//		cout << i << ":" << endl;
//		cout << y_pre << endl; 
//		double sum = 0.0;
//		for(int i = 0; i != y_pre.cols(); i++){
//			if(y(0, i) == 1 && y_pre(0, i) > 0.5 || y(0, i) == 0 && y_pre(0, i) < 0.5)
//				sum++;
//		}
//		cout << sum/y.cols() << endl;
//	}

