# TinyMPC Complete Build Flow
# Usage: make N=<horizon> FREQ=<frequency>

# Parameters
N ?= 5
FREQ ?= 100

# Project paths
PROJ_DIR := $(abspath .)
PROJ_NAME = tinympcproj_N$(N)_$(FREQ)Hz_float
HLS_PROJ = tinympc_N$(N)_$(FREQ)Hz_float
HLS_DIR = $(PROJ_NAME)
IP_DIR = $(PROJ_DIR)/$(HLS_DIR)/$(HLS_PROJ)/solution1/impl/ip
VIVADO_DIR = $(PROJ_NAME)_vivado

# Main targets
.PHONY: all generate hls bitstream clean help

all: bitstream

# Stage 1: Generate HLS code
generate: $(HLS_DIR)/tinympc_solver.cpp

$(HLS_DIR)/tinympc_solver.cpp:
	python3 tinympc_ip_generator.py --N $(N) --freq $(FREQ) --precision float

# Stage 2: HLS synthesis and IP packaging  
hls: $(IP_DIR)/xilinx_com_hls_tinympc_solver_1_0.zip

$(IP_DIR)/xilinx_com_hls_tinympc_solver_1_0.zip: $(HLS_DIR)/tinympc_solver.cpp
	cd $(HLS_DIR) && vitis_hls -f csim.tcl

# Stage 3: Generate bitstream (combined XSA creation and extraction)
bitstream: bitstream/$(PROJ_NAME).bit

bitstream/$(PROJ_NAME).bit: $(IP_DIR)/xilinx_com_hls_tinympc_solver_1_0.zip
	mkdir -p $(VIVADO_DIR)
	cd $(VIVADO_DIR) && PROJ_NAME=$(PROJ_NAME) IP_DIR=$(IP_DIR) vivado -mode batch -source $(PROJ_DIR)/impl/workflow.tcl
	cd $(VIVADO_DIR) && unzip -o $(PROJ_NAME).xsa -d hwFile_$(PROJ_NAME) && \
		mkdir -p ../bitstream && \
		cp -f hwFile_$(PROJ_NAME)/$(PROJ_NAME).bit ../bitstream/$(PROJ_NAME).bit && \
		cp -f hwFile_$(PROJ_NAME)/design_1.hwh ../bitstream/$(PROJ_NAME).hwh

# Clean targets
clean:
	rm -rf $(VIVADO_DIR)
	rm -rf $(HLS_DIR)/$(HLS_PROJ)
	rm -rf bitstream

distclean: clean
	rm -rf $(HLS_DIR)
	rm -rf $(VIVADO_DIR)

# Help
help:
	@echo "TinyMPC Build System"
	@echo "Usage: make N=<horizon> FREQ=<frequency> [target]"
	@echo ""
	@echo "Targets:"
	@echo "  all       - Complete flow (default)"
	@echo "  generate  - Generate HLS code only" 
	@echo "  hls       - Run HLS synthesis"
	@echo "  bitstream - Generate bitstream and hardware files"
	@echo "  clean     - Remove build files"
	@echo "  distclean - Remove all generated files"
	@echo ""
	@echo "Examples:"
	@echo "  make N=5 FREQ=100"
	@echo "  make N=10 FREQ=50 generate"