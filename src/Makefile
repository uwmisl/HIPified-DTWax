HIPCC = hipcc
LIBS = -lz -lmiopen -I$(ROCM_INC)/miopen -L$(ROCM_LIB)

# LIBS = -lhdf5 -lcudnn -I/usr/include/hdf5/serial -L/usr/lib -L/usr/lib/x86_64-linux-gnu/hdf5/serial -I/usr/local/include -L/usr/local/lib -L/usr/local/lib 
# NVCCFLAGS= -std=c++11 --ptxas-options=-v --default-stream per-thread  -O2 -lineinfo -gencode=arch=compute_86,code=sm_86 --compiler-options -Wall -DSDTW -DFP16 -DNO_REF_DEL -DNV_DEBUG 
HIPCFLAGS = -O2 -lineinfo -Wall -DNO_REF_DEL -lwb#-DNV_DEBUG#-DFP16#-DNV_DEBUG #--DFP16 
CXXFLAGS= -fopenmp -march=native -O2#-DNV_DEBUG

# To avoid the warnings about ignoring the return value of functions declared with 'nodiscard' (want the return values checked)
CXXFLAGS += -Wno-unused-result
# export HDF5_PLUGIN_PATH=$(HOME)/.local/hdf5/lib/plugin

# ifdef READ_UNTIL
#     MAIN_FILE = ru_main.cu
# else
#     MAIN_FILE = main.cu
# endif

all: main

main: $(MAIN_FILE)
	$(HIPCC) $(HIPCFLAGS) $(CXXFLAGS) $(LIBS) main.cpp -o main

format:
	clang-format -i *.cpp include/*.hpp include/*.cpp

clean:
		rm -rf main

$(LIBHDF5):
	cd ../hdf5 && \
	./configure --enable-threadsafe --disable-hl --prefix=`pwd` --enable-shared=no --with-pic=yes || exit 255
	make -j 8 -C ../hdf5 && make -C ../hdf5 install
