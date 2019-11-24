#include <iostream>
#include "Eigen/Dense"
#include "net.h"
#include <typeinfo>

using namespace Eigen;
using namespace std;

int main()
{
	Net net;
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
	
	
//	MatrixXd m(2, 3);
//	m << 1, 3, 4, 
//		 4, 3, 1;
//	cout << m << endl;
//	cout << net.sigmoid(m) << endl;
//	
//	
//	MatrixXd n(2,1);
//	
//	n << 2,
//		 3;
//	cout << n << endl;
////	n.resize(2, 3);
//	auto b = m;
////	b.fill(n);
//	cout << b << endl;
//	cout << m.array() + b.array() << endl;
	
	int n = 9;
	net.sum(n);
	cout << n << endl;
	
	
	
	
  	return 0; 
}
