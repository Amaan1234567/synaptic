# CMake generated Testfile for 
# Source directory: /home/amaan/Documents/synaptic
# Build directory: /home/amaan/Documents/synaptic
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(run_all_tests "/home/amaan/Documents/synaptic/tests")
set_tests_properties(run_all_tests PROPERTIES  _BACKTRACE_TRIPLES "/home/amaan/Documents/synaptic/CMakeLists.txt;40;add_test;/home/amaan/Documents/synaptic/CMakeLists.txt;0;")
subdirs("_deps/googletest-build")
