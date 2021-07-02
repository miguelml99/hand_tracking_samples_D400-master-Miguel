#----------------------------------------------------------------
# Generated CMake target import file for configuration "Debug".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "realsense::realsense" for configuration "Debug"
set_property(TARGET realsense::realsense APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(realsense::realsense PROPERTIES
  IMPORTED_IMPLIB_DEBUG "${_IMPORT_PREFIX}/lib/realsense.lib"
  IMPORTED_LOCATION_DEBUG "${_IMPORT_PREFIX}/bin/realsense.dll"
  )

list(APPEND _IMPORT_CHECK_TARGETS realsense::realsense )
list(APPEND _IMPORT_CHECK_FILES_FOR_realsense::realsense "${_IMPORT_PREFIX}/lib/realsense.lib" "${_IMPORT_PREFIX}/bin/realsense.dll" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
