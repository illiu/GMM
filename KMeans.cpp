#include "KMeans.h"

Cluster_KMeans::Cluster_KMeans(int dim)
{
	dimension = dim;
	mean	= VectorXd::Zero(dim);
}

KMeans::KMeans(int n_clust, int dim)
{
	n_cluster = n_clust;
	dimension = dim;

	for ( int i = 0 ; i < n_clust ; i++ )
	{
		clusters.push_back(Cluster_KMeans(dim));
	}

	points = NULL;
}

void KMeans::setSamples(vector<VectorXd> *samples)
{
	if ( points )
		delete points;

	classify.clear();

	points = samples;
	classify.resize(samples->size());
}

void KMeans::iterate(void)
{
	int i, j;

	int pidx;
	vector<VectorXd>::iterator pit;

	vector<VectorXd> temp_clust; 
	vector<int> temp_size;
	temp_clust.resize(clusters.size());
	temp_size.resize(clusters.size());

	for ( i = 0 ; i < clusters.size() ; i++ )
	{
		temp_clust[i]	= VectorXd::Zero(dimension);
		temp_size[i]	= 0;
	}

	for ( pit = points->begin(), pidx = 0 ; pit != points->end() ; pit++, pidx++ )
	{
		double	max_dist2 = 10000.;
		int 	cidx, max_clust = 0;
		vector<Cluster_KMeans>::iterator cit;
		for ( cit = clusters.begin(), cidx = 0 ; cit != clusters.end(); cit++, cidx++ )
		{
			VectorXd dist;
			double dist2;
			dist = cit->mean - (*pit);
			dist2 = dist.transpose() * dist;
			if ( dist2 < max_dist2  )
			{
				max_dist2 = dist2;
				max_clust = cidx;
			}
		}
		classify[pidx] = max_clust;
		temp_clust[max_clust] = temp_clust[max_clust] + *pit;
		temp_size[max_clust]++;
	}

	for ( i = 0 ; i < clusters.size() ; i++ )
	{
		clusters[i].mean = temp_clust[i] / temp_size[i];
		cout << "Cluster " << endl << clusters[i].mean << endl;
	}
}

void KMeans::iterate(int times)
{
	int i;

	for ( i = 0 ; i < times ; i++ )
	{
		iterate();
	}
}

KMeans::~KMeans(void)
{
	if ( points )
		delete points;
}
