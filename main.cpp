#include <iostream>
#include "Eigen/Dense"
#include "net.h"
#include "train.h"

using namespace Eigen;
using namespace std;

int main()
{
	
	Net net(8);
	net.add_lay(17);
	net.add_lay(1);
	
	cout << "需训练的参数 : " << net.get_NUM_PAR() << "个" << endl;
	cout << "共有 : " << net.get_NUM_LAY() << "层" << endl;
	
	MatrixXd x(8,17), y(1,17);

	x << 2,3,3,2,1,2,3,3,3,2,1,1,2,1,3,1,2,
		 2,2,2,2,2,1,1,1,1,3,3,2,1,1,1,2,2,
		 2,3,2,3,2,2,2,2,3,1,1,2,2,3,2,2,3,
		 1,1,1,1,1,1,2,1,2,1,3,3,2,2,1,3,2,
		 3,3,3,3,3,2,2,2,2,1,1,1,3,3,2,1,2,
		 1,1,1,1,1,2,2,1,1,2,1,2,1,1,2,1,1,
		 0.697,0.744,0.634,0.608,0.556,0.403,0.481,0.437,0.666,0.243,0.245,0.343,0.639,0.657,0.360,0.593,0.719,
		 0.460,0.376,0.264,0.318,0.215,0.237,0.149,0.211,0.091,0.267,0.057,0.099,0.161,0.198,0.370,0.042,0.103;
    y << 1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0;
		
	Train tra(x, y);

	cout << endl << "训练前" << endl;
	auto y_pre = net.predict(x, y);
	cout << y_pre << endl; 
	
	Net ne(8);
	ne.add_lay(17);
	ne.add_lay(1);
	
	for(int i = 500; i <= 5000; i += 500){
		net = ne;
		tra.train(net, i);
		auto y_pre = net.predict(x, y);
		cout << i << ":" << endl;
		cout << y_pre << endl; 
		double sum = 0.0;
		for(int i = 0; i != y_pre.cols(); i++){
			if(y(0, i) == 1 && y_pre(0, i) > 0.5 || y(0, i) == 0 && y_pre(0, i) < 0.5)
				sum++;
		}
		cout << sum/y.cols() << endl;
	}

  	return 0; 
}
