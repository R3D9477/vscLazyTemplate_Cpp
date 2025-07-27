if(DEFINED ENV{VCPKG_ROOT})
    set(CMAKE_TOOLCHAIN_FILE "${CMAKE_TOOLCHAIN_FILE_VCPKG}" CACHE STRING "Vcpkg toolchain file")
else()
    set(CMAKE_TOOLCHAIN_FILE "${CMAKE_TOOLCHAIN_FILE_CONAN}")
endif()

message("")
message("--> CMake toolchain is " ${CMAKE_TOOLCHAIN_FILE})
message("")
