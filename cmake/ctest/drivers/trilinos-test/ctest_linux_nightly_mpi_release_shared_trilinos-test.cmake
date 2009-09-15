
INCLUDE("${CTEST_SCRIPT_DIRECTORY}/TrilinosCTestDriverCore.trilinos-test.gcc.cmake")

#
# Set the options specific to this build case
#

SET(COMM_TYPE MPI)
SET(BUILD_TYPE RELEASE)
SET(BUILD_DIR_NAME MPI_RELEASE_10.0_SHARED)
SET(Trilinos_TRACK "Nightly Release 10.0")

SET(Trilinos_BRANCH "-r trilinos-release-10-0-branch")

SET(Trilinos_ENABLE_SECONDARY_STABLE_CODE ON)

SET( EXTRA_CONFIGURE_OPTIONS
  "-DCMAKE_VERBOSE_MAKEFILE:BOOL=TRUE"
  "-DDART_TESTING_TIMEOUT:STRING=120"
  "-DTrilinos_ENABLE_EXPLICIT_INSTANTIATION:BOOL=ON"
  "-DBUILD_SHARED_LIBS:BOOL=ON"
  "-DMPI_BASE_DIR:PATH=/usr/lib64/openmpi/1.2.7-gcc"
  "-D TPL_ENABLE_Pthread:BOOL=ON"
  "-D TPL_ENABLE_Boost:BOOL=ON"
  "-D Boost_INCLUDE_DIRS:FILEPATH=/home/trilinos/tpl/gcc4.1.2/boost_1_38_0"
  )

#
# Set the rest of the system-specific options and run the dashboard build/test
#

TRILINOS_SYSTEM_SPECIFIC_CTEST_DRIVER()
