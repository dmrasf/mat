#include <iostream>
#include "Eigen/Dense"
#include "net.h"
#include "rbf.h"
#include "clustering.h"
#include "train.h"
#include <fstream>
#include <windows.h>
#include <cstdio>
#include <bitset>
#include <iomanip>

using namespace Eigen;
using namespace std;

//预测函数    需要预测的个数 
void predict(Net &net, int n = 10);
//训练函数   训练参数   训练函数    训练次数   一次训练的个数 
void train(Net &net, Train &tra, int n = 1, int m = 5000);
void train(Clustering &clu, Train &tra, int n = 1, int m = 5);
void predict(Clustering &clu, int n = 10);
//读取图片 
bool read_bmp(const char*); 
void data(MatrixXd&, const char*, int m);
int main()
{

//	MatrixXd x(2,16), y(1,16);
//	x << 3,4,3,4,-4,-2,-5,-3,37,23,54,23,-0.3,-0.5,-0.7,-0.6,
//		 4,5,3,2,6,7,4,5,-4,-3,-4,-5,-1,-1,-1.3,-0.9; 
//	y << 0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3;
//	
//	Clustering clu(4, 2);
//	Train tra(x, y);
//	tra.train(clu, 100);
//	auto y_pre = clu.predict(x);
//	cout << "y = " << endl << y << endl;
//	cout << "y_pre = " << endl << y_pre.transpose() << endl;
	
	Clustering clu(10, 784);
	Train tra;
	train(clu, tra, 20, 10000);
	predict(clu, 1500);
	
	
//	read_bmp("5.bmp");
	
//	Rbf r(2,20,2);
//	MatrixXd x(2,10), y(2,10);
////	x.setRandom();
//	x << 0,1,2,3,4,5,6,7,8,9,
//		 3,4,5,6,7,8,9,1,2,3;
//	y = ((x.array().square()+1)*0.1).square();
//	Train tra(x, y);
//	int n = 1000;
//	Train tr;
//	while(n--){
//		tra.train(r,1);
//	}
//
//	cout << "y_tra = " << endl << y << endl;
//	cout << "y_pre = " << endl << r.predict(x) << endl;
	

		
//	Net net(784);
//	net.add_lay(50);
//	net.add_lay(10);
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

void train(Clustering &clu, Train &tra, int n, int m){
	MatrixXd x_train(784, m), y_train(1, m);
	data(x_train, "train_x.csv", m);
	
	ifstream r_train_y("train_y.csv");
	int i = 0;
	string word;
	while(getline(r_train_y, word, ',')){		
		y_train(0, i) = stoi(word);
		i++;
		if(i == m)
			break;
	}
	tra.get_new(x_train, y_train);
	tra.train(clu, n);
}

void predict(Clustering &clu, int n){
	double sum = 0.0;
	MatrixXd x_test(784, n), y_test(1, n);
	data(x_test, "test_x.csv", n);
	auto y_pre = clu.predict(x_test);
	ifstream r_test_y("test_y.csv");
	int i = 0;
	string word;
	while(getline(r_test_y, word, ',')){		
		y_test(0, i) = stoi(word);
		if(y_pre(i) == y_test(0, i))
			sum++;
		i++;
		if(i == n)
			break;
	}
	cout << "y = " << endl << y_test << endl;
	cout << "y_pre = " << endl << y_pre.transpose() << endl;
	cout << "准确率 = " << sum/n << endl;
}

void data(MatrixXd &x, const char *name, int m){
	ifstream r(name);
	string line;
	double train_x_[784];
	int j = 0;
	while(getline(r, line)){
		istringstream record(line);
		string word;
		int i = 0;
		while(getline(record, word, ',')){		
			train_x_[i] = stoi(word);
			i++;
		}
		MatrixXd train_x = Map<Matrix<double, 1, 784>>(train_x_);
		train_x = train_x/255;
		x.col(j) = train_x.transpose();
//		if(f == 1)
//			cout << Map<Matrix<double, 28, 28, RowMajor>>(train_x_) << endl << endl;
		j++;
		if(j == m)
			break;
	}
}

bool read_bmp(const char *name){
	FILE *bmp = fopen(name, "rb");
	
//	BITMAPFILEHEADER file_h;
//	BITMAPINFOHEADER info_h;
//	RGBQUAD rgb;
//		
//	fread(&file_h, sizeof(BITMAPFILEHEADER), 1, bmp);
//	fread(&info_h, sizeof(BITMAPINFOHEADER), 1, bmp);
//	fread(&rgb, sizeof(RGBQUAD), 1, bmp);
//	
//	cout << sizeof(info_h) << endl;
//	cout << rgb.rgbBlue << endl;
//	cout << rgb.rgbReserved <<endl;
//	
//	cout << info_h.biHeight << endl;
//	cout << info_h.biWidth << endl;
//	cout << info_h.biSize << endl;
//	cout << info_h.biClrUsed << endl;
//	cout << info_h.biSizeImage << endl;
//	cout << info_h.biBitCount << endl;
//	cout << info_h.biPlanes << endl;
//	
////	cout << file_h.bfType << endl;
//	cout << file_h.bfSize << endl;
//	cout << file_h.bfOffBits << endl;
//	
//	fseek(bmp, 62, 0);
//	int n[280*36];
//	int m = 0;
//	for(int i = 0; i != 280; i++){
//		for(int j = 0; j != 36; j++){
//			fread(&m, 1, 1, bmp);
//			cout << setw(5) << (~m)+256; 
//			n[i*36+j] = (~m)+256;
//		}
//		cout << endl;
//	}
//	fseek(bmp,0,0);
	FILE *fw = fopen("we.bmp", "ab");
//	fwrite(&bmp,sizeof(BITMAPFILEHEADER),1,fw);  //写入文件头
//	fseek(bmp, 14, 0);
//	fwrite(&bmp,sizeof(BITMAPINFOHEADER),1,fw);
//	fseek(bmp, 54, 0);
//	fwrite(&bmp,sizeof(RGBQUAD),1,fw);
	fwrite(&bmp,1,10142,fw);
//	fwrite(&n, 1, 280*36, fw);
	fclose(fw); 
	
	fclose(bmp);
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
	}
}

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

