#include <iostream>
#include <stdio.h>
#include <math.h>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

#include <FL/Fl.H>
#include <FL/Fl_Double_Window.H>
#include <FL/fl_draw.H>
#include <FL/Fl_Button.H>
#include <FL/Fl_Value_Slider.H>
#include "GMM.h"
#include "KMeans.h"

using namespace std;
using Eigen::MatrixXf;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::VectorXcd;
using Eigen::MatrixXcd;
using Eigen::EigenSolver;

double scale;
double x0;
double yy0;
double y_org;
double dt = 0.001;

static int win_width(400);
static int win_height(400);
static char const * win_title("simul_point_mass");

#define NUM_SAMPLE (1000)
#define NUM_CLUSTER (3)
#define NUM_DIMENSION (2)

class MySim : public Fl_Widget
{
public:
	MySim(int xx, int yy, int width, int height);
	virtual ~MySim();
	
	virtual void draw();

	void tick();
	static void timer_cb(void *param);
	
	int handle(int e)
	{
		int ret = 0;
		switch (e)
		{
		case FL_PUSH:
			double posX, posY;

			posX = Fl::event_x();
			posY =  Fl::event_y();

			fprintf(stderr, "PUSH EVENT!!! (%f,%f)\n", posX, posY);

//			target_pos[0] = (posX - x0) / scale;
//			target_pos[1] = (y_org - posY) / scale;
			ret = 1;

			break;
		case FL_MOUSEWHEEL:
			fprintf(stderr, "MOUSE WHEEL!!! %f\n", scale);
			scale += Fl::event_dy()/10.;

			if ( scale < 0. )
				scale = 0.;
			ret = 1;
			break;
		case FL_RELEASE:
			fprintf(stderr, "RELEASE EVENT!!!\n");
			break;
		}
		Fl_Widget::handle(e);

		return ret;
	}
};

class MyWindow : public Fl_Double_Window 
{
public:
	MyWindow(int width, int height, const char *title);

	virtual void resize(int x, int y, int w, int h);	

	Fl_Button *control;
	Fl_Button *again;
	Fl_Button *quit;
	MySim *sim;
	Fl_Value_Slider *mKpSlider;
	Fl_Value_Slider *mKvSlider;
	static GMM			gmm;
	static KMeans		kmeans;

	static void cb_control(Fl_Widget *widget, void *param);
	static void cb_again(Fl_Widget *widget, void *param);
	static void cb_quit(Fl_Widget *widget, void *param);
	static void cb_kp(Fl_Widget *widget, void *param);
	static void cb_kv(Fl_Widget *widget, void *param);
};

MySim::MySim(int xx, int yy, int width, int height)
: Fl_Widget(xx, yy, width, height, "")
{
	Fl::add_timeout(dt, timer_cb, this);
}


MySim::~MySim()  {
//	Fl::rgmm(timer_cb, this);
}
 
static bool paused_ready = false;
void MySim::
timer_cb(void * param)
{
	static double lastZ = 10.0;

	double x, z;
//	if ( z <= 0.0 && lastZ <= 0.0 )
//		paused = true;
	

	reinterpret_cast<MySim*>(param)->redraw();
/*
	if ( ! paused || ! paused_ready ) {
		reinterpret_cast<Simulator*>(param)->tick();
	}

	if ( paused )
		paused_ready = true;
	if ( ! paused )
		paused_ready = false;
*/
	Fl::repeat_timeout(dt, // gets initialized within tick()
			   timer_cb,
			   param);
}

void MySim::
draw()
{
	double xx, zz;
	double xmax, ymax;

	xmax = w() / 2.;
	ymax = h() / 2.;

	double ratio = min(xmax,ymax) / 5. / (scale + 0.1);

#if 0
	if (w() > h()) {
	  scale = h() / 16.0;
	}
	else {
	  scale = w() / 16.0;
	}
#endif
	x0 = w()/2;
	yy0 = h()/2;
	y_org = h() * 3.0 / 4.0;

	fl_color(FL_BLACK);
	fl_rectf(x(), y(), w(), h());


	fl_color(FL_GREEN);
//	vector<VectorXd> *points = (MyWindow::gmm.points);
	vector<VectorXd> *points = (MyWindow::kmeans.points);
	vector<VectorXd>::iterator it;
	vector<int>::iterator iit;

	if ( points )
	{
		for ( it = points->begin(), iit = MyWindow::kmeans.classify.begin() ; it != points->end() ; it++, iit++ )
		{
			switch ( *iit )
			{
			case 0:
				fl_color(FL_YELLOW);
				break;
			case 1:
				fl_color(FL_RED);
				break;
			case 2:
				fl_color(FL_CYAN);
				break;
			}
			fl_point(x0 + (*it)(0)*ratio,
					 yy0 + (*it)(1)*ratio);
		}
	}

	int c;
	for ( c = 0 ; c < NUM_CLUSTER ; c++ )
	{
		switch (c)
		{
		case 0:
			fl_color(FL_YELLOW);
			break;
		case 1:
			fl_color(FL_RED);
			break;
		case 2:
			fl_color(FL_CYAN);
			break;
		}

#if 1
		Cluster cl = MyWindow::gmm.clusters[c];
		double l;

		fl_rectf(x0 + cl.mean[0]*ratio-2., yy0 + cl.mean[1]*ratio-2., 5., 5.);

		l = cl.eigval[0].real();
		fl_line( x0+ cl.mean[0]*ratio - l*cl.eigvec.col(0)[0].real()*ratio, 
				yy0 + cl.mean[1]*ratio - l*cl.eigvec.col(0)[1].real()*ratio,
				x0+ cl.mean[0]*ratio + l*cl.eigvec.col(0)[0].real()*ratio, 
				yy0 + cl.mean[1]*ratio + l*cl.eigvec.col(0)[1].real()*ratio);
		l = cl.eigval[1].real();
		fl_line( x0+ cl.mean[0]*ratio - l*cl.eigvec.col(1)[0].real()*ratio, 
				yy0 + cl.mean[1]*ratio - l*cl.eigvec.col(1)[1].real()*ratio,
				x0+ cl.mean[0]*ratio + l*cl.eigvec.col(1)[0].real()*ratio, 
				yy0 + cl.mean[1]*ratio + l*cl.eigvec.col(1)[1].real()*ratio);
#else
		Cluster_KMeans cl = MyWindow::kmeans.clusters[c];
		fl_rectf(x0 + cl.mean[0]*ratio-2., yy0 + cl.mean[1]*ratio-2., 5., 5.);
#endif
	}


}


GMM MyWindow::gmm = GMM(NUM_CLUSTER,NUM_DIMENSION);
KMeans MyWindow::kmeans = KMeans(NUM_CLUSTER,NUM_DIMENSION);

MyWindow:: 
MyWindow(int width, int height, const char * title) 
    : Fl_Double_Window(width, height, title)
{ 
    Fl::visual(FL_DOUBLE|FL_INDEX); 
    begin(); 
    sim = new MySim(0, 0, width, height - 40); 
    control = new Fl_Button(5, height - 35, 100, 30, "&Control"); 
    control->callback(cb_control); 
    again = new Fl_Button(width / 2 - 50, height - 35, 100, 30, "&Again"); 
    again->callback(cb_again, this); 
	printf("WIN: %p\n", this);
    quit = new Fl_Button(width - 105, height - 35, 100, 30, "&Quit"); 
    quit->callback(cb_quit, this); 

	mKpSlider = new Fl_Value_Slider( 10, height - 35, width / 4 - 100, 30, "Kp");
	mKpSlider->type(FL_HORIZONTAL);
	mKpSlider->bounds(0., 10.);
	mKpSlider->callback(cb_kp, this);
	mKpSlider->value(0.);

	mKvSlider = new Fl_Value_Slider( width / 4 + 20, height - 35, width / 4 - 100, 30, "Kv");
	mKvSlider->type(FL_HORIZONTAL);
	mKvSlider->bounds(0., 90.);
	mKvSlider->callback(cb_kv, this);
	mKvSlider->value(0.);

    end(); 
    resizable(this); 
	resize(0,0,width, height);
    show(); 
}

void MyWindow::
resize(int x, int y, int w, int h)
{
	Fl_Double_Window::resize(x, y, w, h);
	sim->resize(0, 0, w, h-40);
	mKpSlider->resize(10, h-35, (w-230)/2 - 5,30);
	mKvSlider->resize(15 + (w-230)/2, h-35, (w-230)/2 - 5,30);
	control->resize(w-225, h - 35, 70, 30);
    again->resize(w-150 , h - 35, 70, 30);
	quit->resize(w-75, h-35, 70, 30);

}
  
void MyWindow::
cb_control(Fl_Widget *widget, void *param)
{
	MyWindow *pWin = (MyWindow *)param;
	pWin->gmm.iterate();
	pWin->kmeans.iterate();
}

void MyWindow::
cb_again(Fl_Widget *widget, void *param)
{
	MyWindow *pWin = (MyWindow *)param;
	pWin->gmm.reset();
}

void MyWindow::
cb_quit(Fl_Widget *widget, void *param)
{
	reinterpret_cast<MyWindow*>(param)->hide();
}

void MyWindow::
cb_kp(Fl_Widget *widget, void *param)
{
	Fl_Value_Slider *pSlider = (Fl_Value_Slider *)widget;
	MyWindow *pWin = (MyWindow *)param;
	
//	myModel.tau = value;
}

void MyWindow::
cb_kv(Fl_Widget *widget, void *param)
{
	Fl_Value_Slider *pSlider = (Fl_Value_Slider *)widget;
	MyWindow *pWin = (MyWindow *)param;
	
//	myModel.tau = value;
}

int main(int argc, char* argv[])
{
	int i;
	VectorXd p1(NUM_DIMENSION);
	vector<VectorXd> *points;

	points = new vector<VectorXd>;
	points->clear();

	for (i = 0 ; i < NUM_SAMPLE/2 ; i++ )
	{
		double x, y;

		x = getRandomNormal();
		y = getRandomNormal();

		p1(0) = 2. + x/2.;
		p1(1) = y*2.;

		points->push_back(p1);
	}
	MatrixXd rot(NUM_DIMENSION,NUM_DIMENSION);

	double th = M_PI / 4.;
	rot(0,0) = cos(th);
	rot(0,1) = sin(th);
	rot(1,0) = -sin(th);
	rot(1,1) = cos(th);

	for (i = 0 ; i < NUM_SAMPLE/2 ; i++ )
	{
		double x, y;

		x = getRandomNormal();
		y = getRandomNormal();

		p1(0) = x*3.;
		p1(1) = -2. + y/3.;

		points->push_back(rot*p1);
	}

	for (i = 0 ; i < NUM_SAMPLE/2 ; i++ )
	{
		double x, y;

		x = getRandomNormal();
		y = getRandomNormal();

		p1(0) = -2. + x*3;
		p1(1) = 2. + y/3.;

		points->push_back(p1);
	}


	MyWindow win(win_width, win_height, win_title);

#if 1

	win.gmm.clusters[0].pi = 0.5;
	win.gmm.clusters[1].pi = 0.2;
	win.gmm.clusters[2].pi = 0.3;

	win.gmm.clusters[0].mean[0] = 0.;
	win.gmm.clusters[0].mean[1] = 0.;
	win.gmm.clusters[1].mean[0] = 5.;
	win.gmm.clusters[1].mean[1] = 0.;
	win.gmm.clusters[2].mean[0] = -5.;
	win.gmm.clusters[2].mean[1] = 0.;

	win.kmeans.clusters[0].mean[0] = 0.;
	win.kmeans.clusters[0].mean[1] = 0.;
	win.kmeans.clusters[1].mean[0] = 5.;
	win.kmeans.clusters[1].mean[1] = 0.;
	win.kmeans.clusters[2].mean[0] = -5.;
	win.kmeans.clusters[2].mean[1] = 0.;

	win.gmm.clusters[0].cov = MatrixXd::Identity(2,2);
	win.gmm.clusters[1].cov = MatrixXd::Identity(2,2);
	win.gmm.clusters[2].cov = MatrixXd::Identity(2,2);

	win.gmm.setDefaultClusters();

	win.gmm.setSamples(points);
	win.kmeans.setSamples(points);
//	win.gmm.reset();
#endif

	int ret = Fl::run();

/*
	while ( myModel.z > 0 )
	{
		myModel.update();
		printf("x    : %5f, z    : %5f\n", myModel.x, myModel.z);
		printf("xdot : %5f, zdot : %5f\n", myModel.xdot, myModel.zdot);
		printf("xddot: %5f, zddot: %5f\n", myModel.xddot, myModel.zddot);
	}
*/
	
	return ret;
}
