# CMake generated Testfile for 
# Source directory: C:/Users/marco/source/repos/GP_CPP20_C
# Build directory: C:/Users/marco/source/repos/GP_CPP20_C/out/build/x64-debug
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test([=[pruebas_gp_c]=] "C:/Users/marco/source/repos/GP_CPP20_C/out/build/x64-debug/gp_c.exe" "--test")
set_tests_properties([=[pruebas_gp_c]=] PROPERTIES  _BACKTRACE_TRIPLES "C:/Users/marco/source/repos/GP_CPP20_C/CMakeLists.txt;39;add_test;C:/Users/marco/source/repos/GP_CPP20_C/CMakeLists.txt;0;")
add_test([=[verificacion_cruzada_B_C]=] "C:/Users/marco/source/repos/GP_CPP20_C/out/build/x64-debug/cross_check.exe" "C:/Users/marco/source/repos/GP_CPP20_C/data/ejemplo.csv")
set_tests_properties([=[verificacion_cruzada_B_C]=] PROPERTIES  _BACKTRACE_TRIPLES "C:/Users/marco/source/repos/GP_CPP20_C/CMakeLists.txt;46;add_test;C:/Users/marco/source/repos/GP_CPP20_C/CMakeLists.txt;0;")
