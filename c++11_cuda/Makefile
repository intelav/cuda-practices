

# Location of the CUDA Toolkit
CUDA_PATH ?= /usr/local/cuda

HOST_COMPILER ?= g++
NVCC          := $(CUDA_PATH)/bin/nvcc -ccbin $(HOST_COMPILER)

ifeq ($(TARGET_ARCH),aarch64)
    ifeq ($(TARGET_OS), linux)
        HOST_COMPILER ?= aarch64-linux-gnu-g++
    endif
endif
TARGET_SIZE := 64 # 64-bit machine
# internal flags
NVCCFLAGS   := -m${TARGET_SIZE}
CCFLAGS     :=
LDFLAGS     := -Map
EXTRA_LDFLAGS := output.map
# Debug build flags
ifeq ($(dbg),1)
      NVCCFLAGS += -g -G
      BUILD_TYPE := debug
else
      BUILD_TYPE := release
endif

#SMS ?= 53 61 70 72 75 80 86 87 90
SMS ?= 86 87
GENCODE_FLAGS := 

ifeq ($(GENCODE_FLAGS),)
# Generate SASS code for each SM architecture listed in $(SMS)
$(foreach sm,$(SMS),$(eval GENCODE_FLAGS += -gencode arch=compute_$(sm),code=sm_$(sm)))

# Generate PTX code from the highest SM architecture in $(SMS) to guarantee forward-compatibility
HIGHEST_SM := $(lastword $(sort $(SMS)))
ifneq ($(HIGHEST_SM),)
GENCODE_FLAGS += -gencode arch=compute_$(HIGHEST_SM),code=compute_$(HIGHEST_SM)
endif
endif


ALL_CCFLAGS :=
ALL_CCFLAGS += $(NVCCFLAGS)
ALL_CCFLAGS += $(EXTRA_NVCCFLAGS)
ALL_CCFLAGS += $(addprefix -Xcompiler ,$(CCFLAGS))
ALL_CCFLAGS += $(addprefix -Xcompiler ,$(EXTRA_CCFLAGS))

ALL_LDFLAGS := 
ALL_LDFLAGS += $(ALL_CCFLAGS)
ALL_LDFLAGS += $(addprefix -Xlinker ,$(LDFLAGS))
ALL_LDFLAGS += $(addprefix -Xlinker ,$(EXTRA_LDFLAGS))

INCLUDES  := -I../Common -I/usr/local/cuda/include
LIBRARIES :=


ALL_CCFLAGS += --threads 0 --std=c++11
# Target rules
all: build

build: c++11_cuda
# Compile the target with all prerequisites ($< and $+ required for prerequisites)
c++11_cuda.o:c++11_cuda.cu
	$(EXEC) $(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

c++11_cuda: c++11_cuda.o
	$(EXEC) $(NVCC) $(ALL_LDFLAGS) $(GENCODE_FLAGS) -o $@ $+ $(LIBRARIES)

run: build
	$(EXEC) ./c++11_cuda
clean: 
	rm -f c++11_cuda c++11_cuda.o

clobber: clean