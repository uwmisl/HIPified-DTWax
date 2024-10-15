HIPCC = hipcc
LIBS = -lz -lMIOpen -I$(ROCM_INC)/miopen -L$(ROCM_LIB)

HIPCFLAGS = -O2 -Wall -DNO_REF_DEL
CXXFLAGS= -fopenmp -march=native -O2 # -v (verbose output)

# To avoid the warnings about ignoring the return value of functions declared with 'nodiscard' (want the return values checked)
# CXXFLAGS += -Wno-unused-result

MAIN_FILE = main.cpp

all: main

main: $(MAIN_FILE)
	$(HIPCC) $(HIPCFLAGS) $(CXXFLAGS) $(LIBS) $(MAIN_FILE) -o main

debug: 
	$(HIPCC) $(HIPCFLAGS) -g -O0 $(CXXFLAGS) -v $(LIBS) $(MAIN_FILE) -o main_debug

format:
	clang-format -i *.cpp include/*.hpp include/*.cpp

clean:
	rm -rf main main_debug