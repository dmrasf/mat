#ifndef NET_H
#define NET_H
#include <vector>
#include "Eigen/Dense"
#include <string>
#include <iostream>
#include <fstream> 
using namespace Eigen;
using namespace std;
//�������д�ѵ���Ĳ��� 
class Net{
public:
    Net();
    Net(int);
    Net& operator = (const Net&);
    Net(const Net&);
    virtual bool add_lay(int, const string &fuc = "sigmoid");
    //�������в���  
	int get_NUM_PAR() const;
	//����
	int get_NUM_LAY() const;
	//ÿ��ĺ���
	const string& get_FUNC(int); 
	//Ԥ��
	MatrixXd predict(const MatrixXd&);
	//������̵�������� 
	bool calculate(MatrixXd&, vector<MatrixXd>&, vector<MatrixXd>&); 
	//���²��� ****** 
	void update_w(int, MatrixXd&);
	void update_b(int, VectorXd&);
	//���漰��ȡѵ����Ĳ���
	bool save_par(const string&); 
	bool load_par(const string&);
	//��ȡ���� ���²����� 
	const MatrixXd& get_w(int);
	const VectorXd& get_b(int); 
	//��ͬ�ļ���� ���� 
	MatrixXd sigmoid(const MatrixXd&);
	MatrixXd d_sigmoid(const MatrixXd&); 
	MatrixXd relu(const MatrixXd&);
	MatrixXd d_relu(const MatrixXd&);
	MatrixXd linear(const MatrixXd&);
	MatrixXd d_linear(const MatrixXd&);
	virtual ~Net();
protected://������ɷ��� 
	//��ʼ������ 
    bool add_init_x(int);
    //��  �������� 
    vector<VectorXd> layers;
    //ÿ�����õĺ���
	vector<string> func; 
	//Ȩֵ  
    vector<MatrixXd> weights;
	int Input = -1;
	int Output = -1; 
};

#endif
