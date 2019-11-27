#ifndef NET_H
#define NET_H
#include <vector>
#include "Eigen/Dense"
#include <string>
#include <iostream>
using namespace Eigen;
using namespace std;
//�������д�ѵ���Ĳ��� 
class Net{
public:
    Net();
    Net(int);
    Net& operator = (const Net&);
    Net(const Net&);
    bool add_lay(int, const string &fuc = "sigmoid");
    //�������в��� 
	int get_NUM_PAR() const;
	//����
	int get_NUM_LAY() const;
	//ÿ��ĺ���
	const vector<string>& get_FUNC(); 
	//Ԥ��
	MatrixXd predict(const MatrixXd&);
	//������̵�������� 
	bool calculate(MatrixXd&, vector<MatrixXd>&, vector<MatrixXd>&); 
	
	//���²��� ****** 
	void update_w(int, MatrixXd&);
	void update_b(int, VectorXd&);
	
	//
	const MatrixXd& get_w(int);
	const VectorXd& get_b(int); 
	
	//���� 
	MatrixXd sigmoid(const MatrixXd&);
	MatrixXd relu(const MatrixXd&);
	~Net();
private:
	//��ʼ������ 
    bool add_init_x(int);
    //��  �������� 
    vector<VectorXd> layers;
    //ÿ�����õĺ���
	vector<string> func; 
	//Ȩֵ  
    vector<MatrixXd> weights;
	int Input;
	int Output; 
};

#endif
