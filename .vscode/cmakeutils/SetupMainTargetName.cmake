if(DEFINED ENV{CONTAINER_CONTEXT_FOLDER})
    get_filename_component(MAIN_TARGET $ENV{CONTAINER_CONTEXT_FOLDER} NAME)
else()
    get_filename_component(MAIN_TARGET ${CMAKE_CURRENT_SOURCE_DIR} NAME)
endif()

message("")
message("--> MAIN_TARGET is " ${MAIN_TARGET})
message("")
