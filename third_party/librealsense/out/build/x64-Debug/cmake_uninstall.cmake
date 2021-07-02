IF(NOT EXISTS "C:/Users/migue/Documents/_TFG/Visual Studio/hand_tracking_samples_D400-master/third_party/librealsense/out/build/x64-Debug/install_manifest.txt")
  MESSAGE(WARNING "Cannot find install manifest: \"C:/Users/migue/Documents/_TFG/Visual Studio/hand_tracking_samples_D400-master/third_party/librealsense/out/build/x64-Debug/install_manifest.txt\"")
  MESSAGE(STATUS "Uninstall targets will be skipped")
ELSE(NOT EXISTS "C:/Users/migue/Documents/_TFG/Visual Studio/hand_tracking_samples_D400-master/third_party/librealsense/out/build/x64-Debug/install_manifest.txt")
  FILE(READ "C:/Users/migue/Documents/_TFG/Visual Studio/hand_tracking_samples_D400-master/third_party/librealsense/out/build/x64-Debug/install_manifest.txt" files)
  STRING(REGEX REPLACE "\n" ";" files "${files}")
  FOREACH(file ${files})
    MESSAGE(STATUS "Uninstalling \"$ENV{DESTDIR}${file}\"")
    IF(EXISTS "$ENV{DESTDIR}${file}")
	  EXEC_PROGRAM(
	    "C:/Program Files (x86)/Microsoft Visual Studio/2019/Community/Common7/IDE/CommonExtensions/Microsoft/CMake/CMake/bin/cmake.exe" ARGS "-E remove \"$ENV{DESTDIR}${file}\""
	    OUTPUT_VARIABLE rm_out
	    RETURN_VALUE rm_retval
	    )
	  IF(NOT "${rm_retval}" STREQUAL 0)
	    MESSAGE(FATAL_ERROR "Problem when removing \"$ENV{DESTDIR}${file}\"")
	  ENDIF(NOT "${rm_retval}" STREQUAL 0)
    ELSE(EXISTS "$ENV{DESTDIR}${file}")
	  MESSAGE(STATUS "File \"$ENV{DESTDIR}${file}\" does not exist.")
    ENDIF(EXISTS "$ENV{DESTDIR}${file}")
  ENDFOREACH(file)
ENDIF(NOT EXISTS "C:/Users/migue/Documents/_TFG/Visual Studio/hand_tracking_samples_D400-master/third_party/librealsense/out/build/x64-Debug/install_manifest.txt")
