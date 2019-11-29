#ifndef RBF_H
#define RBF_H
#include <vector>
#include "Eigen/Dense"
#include <string>
#include <iostream>
#include <fstream> 
using namespace Eigen;
using namespace std;

class Rbf
{
public:
	Rbf();
	//��������  ����˸���  ������� 
	Rbf(int, int, int);
    Rbf& operator = (const Rbf&);
    Rbf(const Rbf&);
    void resize(int, int, int);
    //�ҵ��������
	int find_min(const VectorXd&); 
    MatrixXd predict(const MatrixXd&);
    //�������� 
    const MatrixXd& get_c();
    const VectorXd& get_b();
	const MatrixXd& get_w();
	void update_c(const MatrixXd&);
	void update_b(const VectorXd&);
	void update_w(const MatrixXd&);
    bool calculate(MatrixXd&, vector<MatrixXd>&);
    //��˹�˺��� 
	VectorXd Gaussian_kernel(const VectorXd&);
	VectorXd len(const VectorXd&);
	~Rbf();
private:
	//���뵽��������� 
	MatrixXd c;
	//��˹�˺�����Ȩ 
	VectorXd b;
	//���㵽�����Ȩֵ 
	MatrixXd w;
};

#endif
