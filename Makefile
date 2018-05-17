#默认情况下，L1是被开启的，-Xptxas -dlcm=cg可以用来禁用L1

#compile parameters

CC = g++
NVCC = nvcc -arch=sm_35 -lcudadevrt -rdc=true -G --ptxas-options=-v
#NVCC = nvcc -arch=sm_35 -lcudadevrt -rdc=true -G -Xcompiler -rdynamic -lineinfo
#CFLAGS = -g -c #-fprofile-arcs -ftest-coverage -coverage #-pg
#EXEFLAG = -g #-fprofile-arcs -ftest-coverage -coverage #-pg #-O2
#NVCC = nvcc -arch=sm_35 -lcudadevrt -rdc=true 
#CFLAGS = -c #-fprofile-arcs -ftest-coverage -coverage #-pg
CFLAGS = -c -O2 #-fprofile-arcs -ftest-coverage -coverage #-pg
EXEFLAG = -O2 #-fprofile-arcs -ftest-coverage -coverage #-pg #-O2
# TODO: try -fno-builtin-strlen -funswitch-loops -finline-functions

#add -lreadline -ltermcap if using readline or objs contain readline
library = #-lgcov -coverage

objdir = ./objs/
objfile = $(objdir)graph_io.o $(objdir)Timer.o $(objdir)plan.o $(objdir)explore.o

all: gpsm.exe

gpsm.exe: $(objfile) kernel.cu util/cutil.h util/graph_io.h util/large_node.cuh util/gutil.h util/Timer.h
	$(NVCC) $(EXEFLAG) -o gpsm.exe kernel.cu $(objfile)

$(objdir)graph_io.o: util/graph_io.h util/graph_io.cu util/gutil.h util/graph.h
	$(NVCC) $(CFLAGS) util/graph_io.cu -o $(objdir)graph_io.o

$(objdir)Timer.o: util/Timer.cpp util/Timer.h
	$(CC) $(CFLAGS) util/Timer.cpp -o $(objdir)Timer.o

$(objdir)plan.o: init/plan.cu init/plan.h util/graph.h util/gutil.h
	$(NVCC) $(CFLAGS) init/plan.cu -o $(objdir)plan.o

$(objdir)explore.o: filter/explore.cu filter/explore.h util/gutil.h util/graph.h
	$(NVCC) $(CFLAGS) filter/explore.cu -o $(objdir)explore.o

.PHONY: clean dist tarball test sumlines

clean:
	rm -f $(objdir)*
dist: clean
	rm -f *.txt *.exe

tarball:
	tar -czvf gsm.tar.gz main util match io graph Makefile README.md objs

test: main/test.o $(objfile)
	$(CC) $(EXEFLAG) -o test main/test.cpp $(objfile) $(library)

sumline:
	bash script/sumline.sh

