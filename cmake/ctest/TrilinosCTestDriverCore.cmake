# This file is provided solely for backwards compatibility with
# existing driver CTest scripts.

#
# Include the real TriBITS driver script.
#
get_filename_component(CMAKE_CURRENT_LIST_DIR "${CMAKE_CURRENT_LIST_FILE}" PATH)
INCLUDE("${CMAKE_CURRENT_LIST_DIR}/../tribits/ctest/TribitsCTestDriverCore.cmake")