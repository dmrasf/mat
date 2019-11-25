#include <iostream>
#include "Eigen/Dense"
#include "net.h"
#include "train.h"
#include <typeinfo>

using namespace Eigen;
using namespace std;

int main()
{
//	Net net;
//	net.add_lay(3);
//	net.add_lay(1, "softmax");
//	
//	cout << net.get_NUM_LAY() << endl;
//	cout << net.get_NUM_PAR() << endl;
//	auto func = net.get_FUNC();
//	func.push_back("dwc");
//	for(auto s : func)
//		cout << s << endl;
//		
//	auto fun = net.get_FUNC();
//	for(auto s : fun)
//		cout << s << endl;
	
	
//	MatrixXd m(3, 3);
//	m << 1, 3, 4,
//		 2, 4, 4, 
//		 4, 3, 1;
//	cout << m << endl << endl;
////	cout << net.sigmoid(m) << endl;
//	VectorXd n(3);
//	n << 1,
//		 1,
//		 1;
//	
//	for(int i = 0; i != m.cols(); i++)
//		m.col(i) = m.col(i) + n;
//			
////	m.row(1) = m.row(1).array() + 1;
//	
//	cout << m << endl; 
	
	Net net;
	net.add_lay(3);
	net.add_lay(1);
	
	cout << "��ѵ���Ĳ��� : " << net.get_NUM_PAR() << "��" << endl;
	cout << "���� : " << net.get_NUM_LAY() << "��" << endl;
	
	MatrixXd x(1, 3), y(1, 3);
	x << 1, 2, 3;
	y << 0.5, 1, 1.5;
	
//	Train tra(x, y);
//	
//	tra.calculate(net);
//	
//	tra.show_cal();
//	
//	net.predict(x, y); 
//	
	
//	int n = 9;
//	net.sum(n);
//	cout << n << endl;
	
	cout << 5*x << endl; 
	
	
  	return 0; 
}
