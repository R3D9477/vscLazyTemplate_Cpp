#------------------------------------------------------------------------------------------------------
# Add project's libraries

add_subdirectory("lib/libcppinfo")
add_subdirectory("lib/libproject_version")
add_subdirectory("lib/libzexample")

#------------------------------------------------------------------------------------------------------
# Add Boost if required

#target_add_boost(${MAIN_TARGET})
