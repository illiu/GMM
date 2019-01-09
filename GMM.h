#include <iostream>
#include <stdio.h>
#include <math.h>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::VectorXcd;
using Eigen::MatrixXcd;


class Cluster
{
public:
	Cluster(int dimension);
	void update(void);

	int dimension;
	double pi;
	VectorXd mean;
	MatrixXd cov;
	VectorXcd eigval;
	MatrixXcd eigvec;
};

class GMM
{
	public:
	GMM(int n_clust, int dim);
	~GMM(void);
	void reset(void);
	void iterate(void);
	vector<Cluster> clusters;
	void setSamples(vector<VectorXd> *samples);
	void setDefaultClusters();

	vector<VectorXd> *points;

	private:
	int n_cluster;
	int dimension;
	vector<Cluster> default_clusters;
};

double	getRandomNormal(void);
