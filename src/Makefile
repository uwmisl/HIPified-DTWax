HIPCC = hipcc
CXXFLAGS= -fopenmp -fno-omit-frame-pointer -march=native -I$(ROCM_INC)/miopen -I$(ROCM_LIB)/llvm/include
# OpenMP libraries are found in the llvm directory (not standard ROCM_LIB)
LDFLAGS = -Wl,-rpath=$(ROCM_LIB)/llvm/lib
LIBS = -lz -lMIOpen -L$(ROCM_LIB) -L$(ROCM_LIB)/llvm/lib -lomp

HIPCFLAGS = -O2 -Wall

# To avoid the warnings about ignoring the return value of functions declared with 'nodiscard' (want the return values checked)
# CXXFLAGS += -Wno-unused-result
# To avoid the warnings about unused parameters
CXXFLAGS += -Wno-unused-parameter

MAIN_FILE = main.cpp

all: main

main: $(MAIN_FILE)
	$(HIPCC) $(HIPCFLAGS) $(CXXFLAGS) $(LIBS) $(LDFLAGS) $(MAIN_FILE) -o main

debug: 
	$(HIPCC) $(HIPCFLAGS) -Wextra -Werror -g -O0 $(CXXFLAGS) -v $(LIBS) $(LDFLAGS) $(MAIN_FILE) -o main_debug

format:
	clang-format -i *.cpp include/*.hpp include/*.cpp

clean:
	rm -rf main main_debug