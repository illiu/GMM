#include <iostream>
#include <stdio.h>
#include <math.h>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

#include "GMM.h"

using namespace std;
using Eigen::MatrixXf;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::VectorXcd;
using Eigen::MatrixXcd;
using Eigen::EigenSolver;

double	p(VectorXd x, VectorXd mean, MatrixXd cov);

GMM::GMM(int n_clust, int dim)
{
	n_cluster = n_clust;
	dimension = dim;

	for ( int i = 0 ; i < n_clust ; i++ )
	{
		clusters.push_back(Cluster(dim));
	}

	points = NULL;
}

GMM::~GMM(void)
{
	default_clusters.clear();
	clusters.clear();
	if ( points )
	{
		points->clear();
		delete points;
	}
}

void GMM::setSamples(vector<VectorXd> *samples)
{
	if ( points )
	{
		points->clear();
		delete points;
	}
	points = samples;

	reset();
}

void GMM::reset(void)
{
	clusters.clear();
	clusters = default_clusters;
	n_cluster = clusters.size();

	vector<Cluster>::iterator it;

	for ( it = clusters.begin() ; it != clusters.end() ; it++ )
	{
		it->update();
	}

	return;
}

void GMM::setDefaultClusters()
{
	default_clusters.clear();
	default_clusters = clusters;

	return;

}

void GMM::iterate(void)
{
	int i, c;
	int numSamples;
	int numClusters;
	MatrixXd ric;

	if ( !points || points->size() == 0 )
		return;

	numSamples	= points->size();
	numClusters	= clusters.size();
	
	ric = MatrixXd::Zero(numSamples,numClusters);

	for ( i = 0 ; i < numSamples ; i++ )
	{
		double sum = 0.;
		vector<Cluster>::iterator cit;
		for ( cit = clusters.begin(), c = 0 ; cit != clusters.end() ; cit++, c++ )
		{
			ric(i,c) = cit->pi * p((*points)[i], cit->mean, cit->cov);
			sum += ric(i,c);
		}
		if ( sum < 0.01 )
		{
			fprintf(stderr, "Too small ric(%d) %f %f\n", i, ric(i,0), ric(i,1));
		}
		for ( c = 0 ; c < numClusters ; c++ )
		{
			ric(i,c) = ric(i,c) / sum;
		}
	}

	VectorXd mc;
	
	mc = VectorXd::Zero(numClusters);
	for ( i = 0 ; i < numSamples ; i++ )
	{
		for ( c = 0 ; c < numClusters ; c++ )
		{
			mc[c] += ric(i,c);
		}
	}
	fprintf(stderr, "MC %f, %f\n", mc[0], mc[1]);

	clusters[0].pi = mc[0] / (mc[0] + mc[1]);
	clusters[1].pi = mc[1] / (mc[0] + mc[1]);

	fprintf(stderr, "PI %f, %f\n", clusters[0].pi, clusters[1].pi);

	for ( c = 0 ; c < numClusters ; c++ )
	{
		clusters[c].mean = VectorXd::Zero(dimension);

		for ( i = 0 ; i < numSamples ; i++ )
		{
			clusters[c].mean += ric(i,c) * (*points)[i];
		}
		clusters[c].mean = clusters[c].mean / mc[c];
		cerr << "Mean (" << c << ")"<< endl << clusters[c].mean << endl;

		clusters[c].cov = MatrixXd::Zero(dimension,dimension);
		for ( i = 0 ; i < numSamples ; i++ )
		{
			VectorXd dx;
			dx = (*points)[i] - clusters[c].mean;
			clusters[c].cov += ric(i,c) * dx * dx.transpose();
		}
		clusters[c].cov = clusters[c].cov / mc[c];
		cerr << "Cov (" << c << ")"<< endl << clusters[c].cov << endl;

		VectorXcd eig = clusters[c].cov.eigenvalues();
		cerr << "Eigen " << endl << eig << endl;

		clusters[c].update();
	}
}

Cluster::Cluster(int dim)
{
	dimension = dim;
	mean	= VectorXd::Zero(dim);
	cov		= MatrixXd::Zero(dim,dim);
	eigval	= VectorXcd::Zero(dim);
	eigvec	= MatrixXcd::Zero(dim,dim);
}

void Cluster::update(void)
{
	EigenSolver<MatrixXd> es(cov);
	cerr << "EigenVector0 " << endl << es.eigenvectors().col(0) << endl;
	cerr << "EigenVector1 " << endl << es.eigenvectors().col(1) << endl;
	eigval = es.eigenvalues();
	eigvec = es.eigenvectors();
}

double p(VectorXd x, VectorXd mean, MatrixXd Cov)
{
	double ret;
	VectorXd X;

	X = x - mean;

	double exp_value = -0.5 * (X.transpose() * Cov.inverse() * X)(0);

	ret = 1. / (2.*M_PI*sqrt(Cov.determinant()) ) * exp( exp_value );

//	assert( ret >= 0. );

	return ret;
}

double getRandomNormal(void)
{
	double z0, z1;
	double u1, u2;
	double two_pi = 2*M_PI;
	do
	{
		u1 = rand() * (1.0 / RAND_MAX);
		u2 = rand() * (1.0 / RAND_MAX);
	}
	while ( u1 <= 0.0001 );
		
	z0 = sqrt(-2.0 * log(u1)) * cos(two_pi * u2);
	z1 = sqrt(-2.0 * log(u1)) * sin(two_pi * u2);

	return z0;
			
}
