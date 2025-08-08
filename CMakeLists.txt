cmake_minimum_required(VERSION 3.25)

#------------------------------------------------------------------------------------------------------

include("${CMAKE_SOURCE_DIR}/.vscode/cmakeutils/SetupToolchain.cmake")
include("${CMAKE_SOURCE_DIR}/.vscode/cmakeutils/SetupMainTargetName.cmake")

#------------------------------------------------------------------------------------------------------

project(${MAIN_TARGET} VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_COLOR_DIAGNOSTICS ON)

set(CMAKE_CXX_FLAGS_DEBUG "-gdwarf-4")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

#------------------------------------------------------------------------------------------------------
# CUDA

set(CMAKE_CXX_COMPILER_ID "NVIDIA") # workaround to force Cmake detect compiler ID corrently
enable_language(CUDA)

set(CMAKE_CUDA_ARCHITECTURES 60) # Pascal

set(CMAKE_CXX_FLAGS_DEBUG "-g")

#------------------------------------------------------------------------------------------------------

# - DEFINE THE LIST OF THE SOURCES (CURRENT TARGET) BELOW IN VARIABLE SCAN_SOURCES_FOLDER
# - ATTACH EXTERNAL LIBRARIES AT THE BOTTOM OF THAT FILE
list(APPEND SCAN_SOURCES_FOLDER "src")

#------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------

include("${CMAKE_SOURCE_DIR}/.vscode/cmakeutils/Utils.cmake")

#------------------------------------------------------------------------------------------------------

if(NOT TARGET ${MAIN_TARGET})
    add_executable(${MAIN_TARGET} ${${PROJECT_NAME}_SOURCES}) # Setup regular application
endif()

target_include_directories(${MAIN_TARGET} PUBLIC ${${PROJECT_NAME}_INCLUDES})

#------------------------------------------------------------------------------------------------------

include("${CMAKE_SOURCE_DIR}/.vscode/cmakeutils/Cuda.cmake") # Setup CUDA-specific settings

include("${CMAKE_SOURCE_DIR}/.vscode/cmakeutils/SafetyFlags.cmake") # Set compiler's safety flags

#------------------------------------------------------------------------------------------------------

set(EXT_LIBS_FILE "${CMAKE_SOURCE_DIR}/CMakeLists.txt_a_Project_Libs.cmake")
if(EXISTS "${EXT_LIBS_FILE}")
    include("${EXT_LIBS_FILE}") # ATTACH LIBRARIES
endif()

set(EXT_LIBS_FILE "${CMAKE_SOURCE_DIR}/CMakeLists.txt_b_Project_Tests.cmake")
if(EXISTS "${EXT_LIBS_FILE}")
    include("${EXT_LIBS_FILE}") # ADD TESTS
endif()
