#include "train.h"



Train::Train(){
	
}

Train::Train(const MatrixXd &x, const MatrixXd &y){
	x_train = x;
	y_train = y;
} 

bool Train::calculate(Net &net){
	net.calculate(x_train, z, e);
} 

bool Train::train(Net &net, int n){
	while(n > 0){
		n--;
		if(calculate(net))
			for(int i = )
		else
			break;
	} 
}

Train::~Train(){
}



