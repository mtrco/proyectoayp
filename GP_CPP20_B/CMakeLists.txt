cmake_minimum_required(VERSION 3.20)
project(GP_Implementacion_B VERSION 1.0 LANGUAGES CXX)

add_executable(gp_b src/main.cpp src/gp.cpp)
target_include_directories(gp_b PRIVATE include)
target_compile_features(gp_b PRIVATE cxx_std_20)

if(MSVC)
    target_compile_options(gp_b PRIVATE /W4 /permissive-)
    # Release: /O2 ya viene habitualmente habilitado por CMake/MSVC.
    target_compile_options(gp_b PRIVATE $<$<CONFIG:Release>:/O2> $<$<CONFIG:Release>:/GL>)
    target_link_options(gp_b PRIVATE $<$<CONFIG:Release>:/LTCG> $<$<CONFIG:Release>:/OPT:REF> $<$<CONFIG:Release>:/OPT:ICF>)
else()
    target_compile_options(gp_b PRIVATE -Wall -Wextra -Wpedantic)
    target_compile_options(gp_b PRIVATE $<$<CONFIG:Release>:-O3> $<$<CONFIG:Release>:-flto>)
    target_link_options(gp_b PRIVATE $<$<CONFIG:Release>:-flto>)
endif()

enable_testing()
add_test(NAME pruebas_gp_b COMMAND gp_b --test)

# Ejecutar con la carpeta principal como directorio de trabajo (dataset de ejemplo).
set_target_properties(gp_b PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")

add_custom_command(TARGET gp_b POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${CMAKE_SOURCE_DIR}/data"
        "$<TARGET_FILE_DIR:gp_b>/data"
    COMMENT "Copiando archivos de datos..."
)
