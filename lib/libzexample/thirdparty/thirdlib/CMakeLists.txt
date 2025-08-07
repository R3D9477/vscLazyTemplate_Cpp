
#------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------
# THIS LIBRARY

get_filename_component(LIB_NAME ${CMAKE_CURRENT_SOURCE_DIR} NAME) # use current folder name as name of this one library

get_sources_from_dir(${LIB_NAME}_SOURCES ${LIB_NAME}_INCLUDES "include")
get_sources_from_dir(${LIB_NAME}_SOURCES ${LIB_NAME}_INCLUDES "src")

add_library(${LIB_NAME} ${${LIB_NAME}_SOURCES})

target_include_directories(${LIB_NAME} PUBLIC ${${LIB_NAME}_INCLUDES})

target_compile_options(${LIB_NAME} PRIVATE ${SAFETY_FLAGS})

#------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------
# ATTACH EXTERNAL LIBRARIES BELOW...

#...
#...
#...

#------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------
# ADD TESTS BELOW...

target_add_tests_from_folder(${LIB_NAME} "test")

#...
#...
#...
