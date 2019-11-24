#ifndef NET_H
#define NET_H

#include <vector>
#include "Eigen/Dense"

using namespace Eigen;
using namespace std;



//�������д�ѵ���Ĳ��� 
class Net{
public:
    Net();
    Net(int);
    bool add_lay(int, const string &fuc = "sigmoid");
    //�������в��� 
	int get_NUM_PAR() const;
	//����
	int get_NUM_LAY() const;
	//ÿ��ĺ���
	const vector<string>& get_FUNC(); 
	//Ԥ��
	double predict(const MatrixXd&, const MatrixXd&);
	//������̵�������� 
	bool calculate(MatrixXd&, vector<MatrixXd>&, vector<MatrixXd>&); 
	//���� 
	MatrixXd sigmoid(const MatrixXd&);
	MatrixXd relu(const MatrixXd&);

	~Net();
private:
	//��ʼ������ 
    bool add_init_x(int);
    //��  �������� 
    vector<VectorXd> l;
    //ÿ�����õĺ���
	vector<string> func; 
	//Ȩֵ  
    vector<MatrixXd> w;
	int Input;
	int Output; 
};

#endif
