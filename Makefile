cflags= -O2 `pkg-config eigen3 --cflags`\
		`pkg-config xft --cflags --libs`\
		`fltk-config --cxxflags --ldstaticflags`

gmm: main.cpp GMM.cpp KMeans.cpp
	g++ -o $@ $^ $(cflags)

clean:
	rm gmm
