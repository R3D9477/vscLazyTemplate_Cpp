#------------------------------------------------------------------------------------------------------
# Compiler's safety flags loader.

string(TOUPPER ${CMAKE_CXX_COMPILER_ID} SFF_POSTFIX)
set(SAFETY_FLAGS_FILE "${CMAKE_SOURCE_DIR}/CMakeLists.txt_c_Project_SafetyFlags_${SFF_POSTFIX}.cmake")
if(EXISTS "${SAFETY_FLAGS_FILE}")
    include("${SAFETY_FLAGS_FILE}")
endif()

target_compile_options(${MAIN_TARGET} PRIVATE ${SAFETY_FLAGS})

message("")
message("--> Safety flags file is " ${SAFETY_FLAGS_FILE})
message("--> Safety flags: " ${SAFETY_FLAGS})
message("")
