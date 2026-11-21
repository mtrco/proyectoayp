cmake_minimum_required(VERSION 3.20)
project(GP_Implementacion_B VERSION 1.0 LANGUAGES CXX)

add_executable(gp_b src/main.cpp src/gp.cpp)
target_include_directories(gp_b PRIVATE include)
target_compile_features(gp_b PRIVATE cxx_std_20)

option(GP_AVX2 "Compilar con instrucciones AVX2 (/arch:AVX2 o -mavx2)" OFF)

if(MSVC AND CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    # clang-cl (Clang de Visual Studio): misma biblioteca estandar que MSVC, mismas corridas.
    # -ffp-contract=off: sin fusionar multiplicacion y suma (FMA), para que el MSE siga
    # siendo identico bit a bit al de A y C.
    target_compile_options(gp_b PRIVATE /W4 /clang:-ffp-contract=off $<$<CONFIG:Release>:/clang:-O3>)
    if(GP_AVX2)
        target_compile_options(gp_b PRIVATE /arch:AVX2)
    endif()
elseif(MSVC)
    target_compile_options(gp_b PRIVATE /W4 /permissive-)
    # Release: /O2 ya viene habitualmente habilitado por CMake/MSVC.
    target_compile_options(gp_b PRIVATE $<$<CONFIG:Release>:/O2> $<$<CONFIG:Release>:/GL>)
    target_link_options(gp_b PRIVATE $<$<CONFIG:Release>:/LTCG> $<$<CONFIG:Release>:/OPT:REF> $<$<CONFIG:Release>:/OPT:ICF>)
    if(GP_AVX2)
        target_compile_options(gp_b PRIVATE /arch:AVX2)
    endif()
else()
    target_compile_options(gp_b PRIVATE -Wall -Wextra -Wpedantic)
    target_compile_options(gp_b PRIVATE $<$<CONFIG:Release>:-O3> $<$<CONFIG:Release>:-flto>)
    target_link_options(gp_b PRIVATE $<$<CONFIG:Release>:-flto>)
    target_compile_options(gp_b PRIVATE -ffp-contract=off)
    if(GP_AVX2)
        target_compile_options(gp_b PRIVATE -mavx2)
    endif()
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
