cmake_minimum_required(VERSION 3.20)
project(GP_Implementacion_C VERSION 1.0 LANGUAGES CXX)

option(GPC_NATIVE "Optimizar para el procesador local (-march=native / /arch:AVX2)" OFF)
# Ruta a la implementacion B; solo se usa para la herramienta de verificacion cruzada.
set(GP_B_DIR "${CMAKE_SOURCE_DIR}/../GP_CPP20_B" CACHE PATH "Carpeta de la implementacion B")

function(gpc_flags target)
    target_compile_features(${target} PRIVATE cxx_std_20)
    if(MSVC)
        # /fp:precise (por defecto) no fusiona multiplicacion+suma (FMA): resultados
        # bit a bit iguales a B.
        target_compile_options(${target} PRIVATE /W4 /permissive- /fp:precise
            $<$<CONFIG:Release>:/O2> $<$<CONFIG:Release>:/GL>)
        target_link_options(${target} PRIVATE $<$<CONFIG:Release>:/LTCG>
            $<$<CONFIG:Release>:/OPT:REF> $<$<CONFIG:Release>:/OPT:ICF>)
        if(GPC_NATIVE)
            target_compile_options(${target} PRIVATE /arch:AVX2)
        endif()
    else()
        # -ffp-contract=off: sin FMA implicito, para comparar MSE bit a bit con B.
        # -fno-math-errno: permite vectorizar sqrt (no cambia resultados).
        target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic
            -ffp-contract=off -fno-math-errno
            $<$<CONFIG:Release>:-O3> $<$<CONFIG:Release>:-flto>)
        target_link_options(${target} PRIVATE $<$<CONFIG:Release>:-flto>)
        if(GPC_NATIVE)
            target_compile_options(${target} PRIVATE -march=native)
        endif()
    endif()
endfunction()

add_executable(gp_c src/main.cpp src/gpc.cpp)
target_include_directories(gp_c PRIVATE include)
gpc_flags(gp_c)
set_target_properties(gp_c PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")

enable_testing()
add_test(NAME pruebas_gp_c COMMAND gp_c --test)

if(EXISTS "${GP_B_DIR}/src/gp.cpp" AND EXISTS "${GP_B_DIR}/include/gp.hpp")
    add_executable(cross_check tools/cross_check.cpp src/gpc.cpp "${GP_B_DIR}/src/gp.cpp")
    target_include_directories(cross_check PRIVATE include "${GP_B_DIR}/include")
    gpc_flags(cross_check)
    set_target_properties(cross_check PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")
    add_test(NAME verificacion_cruzada_B_C COMMAND cross_check "${CMAKE_SOURCE_DIR}/data/ejemplo.csv")
    message(STATUS "Verificacion cruzada habilitada con B en: ${GP_B_DIR}")
else()
    message(STATUS "No se encontro B en GP_B_DIR=${GP_B_DIR}; cross_check no se compilara")
endif()

add_custom_command(TARGET gp_c POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${CMAKE_SOURCE_DIR}/data" "$<TARGET_FILE_DIR:gp_c>/data"
    COMMENT "Copiando archivos de datos...")
