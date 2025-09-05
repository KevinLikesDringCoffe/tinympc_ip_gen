set PROJ_NAME $::env(PROJ_NAME)
set IP_DIR $::env(IP_DIR)

set PROJ $PROJ_NAME

set BOARD "xczu3eg-sbva484-1-e"

# Create project, if project already exists, it will be deleted
create_project -force $PROJ -part $BOARD

# Add the generated HLS IP
set ip_repo $IP_DIR
set_property ip_repo_paths $ip_repo [current_project]
update_ip_catalog

# Source the block design Tcl script
source ../impl/ultra96_bd.tcl

# Create HDL wrappers
make_wrapper -files [get_files ${PROJ}.srcs/sources_1/bd/${design_name}/${design_name}.bd] -top
add_files -norecurse ${PROJ}.srcs/sources_1/bd/${design_name}/hdl/${design_name}_wrapper.v

# Implement the design
launch_runs impl_1 -to_step write_bitstream -jobs 40

wait_on_run impl_1

write_hw_platform -fixed -include_bit -force -file ${PROJ}.xsa

exit