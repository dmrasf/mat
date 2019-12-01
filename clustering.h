#ifndef CLUSTERING_H
#define CLUSTERING_H
#include <vector>
#include "Eigen/Dense"
#include <string>
#include <iostream>
#include <fstream> 
using namespace Eigen;
using namespace std;

class Clustering
{
public:
//	Clustering();
	Clustering(const MatrixXd&, const VectorXd&);
	Clustering(int, int);
	void train(const MatrixXd&, const MatrixXd&, int, const string&);
	void k_means(const MatrixXd&, const MatrixXd&, int);
	void lvq(const MatrixXd&, const MatrixXd&, int);
	VectorXd predict(const MatrixXd&);
	int find_min(const VectorXd&);
	~Clustering();
private:
	MatrixXd clas;
	VectorXd k;
};

#endif
