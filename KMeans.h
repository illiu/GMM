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


class Cluster_KMeans
{
public:
	Cluster_KMeans(int dimension);
	void update(void);

	int dimension;
	VectorXd mean;
	VectorXd getMean();
	MatrixXd getCov();

private:
	MatrixXd cov_inv;
	double sqrt_cov_det;
};

class KMeans
{
	public:
	KMeans(int n_clust, int dim);
	~KMeans(void);
	void reset(void);
	void iterate(void);
	void iterate(int times);
	void converge(double threshold);
	void setSamples(vector<VectorXd> *samples);
	void setDefaultClusters();
	void update(void);

	vector<Cluster_KMeans>	clusters;
	vector<VectorXd> 		*points;
	vector<int>				classify;

	private:
	int n_cluster;
	int dimension;
	vector<Cluster_KMeans> default_clusters;
};

