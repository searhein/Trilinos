#
# Build all Secondary Stable Trilinos packages in core Trilinos with Intel 12.0.4 compiler
#

INCLUDE("${CTEST_SCRIPT_DIRECTORY}/TribitsCTestDriverCore.pu241.icpc.12.0.4.cmake")
INCLUDE("${CTEST_SCRIPT_DIRECTORY}/SubmitToTrilinos.cmake")

SET(COMM_TYPE SERIAL)
SET(BUILD_TYPE RELEASE)
SET(BUILD_DIR_NAME SERIAL_RELEASE_ICPC_TRILINOS)
#SET(CTEST_TEST_TIMEOUT 900)
SET(Trilinos_ENABLE_SECONDARY_STABLE_CODE ON)
SET(EXTRA_CONFIGURE_OPTIONS
  ${EXTRA_CONFIGURE_OPTIONS}
  -DTeuchos_ENABLE_STACKTRACE:BOOL=ON
  )

TRILINOS_SYSTEM_SPECIFIC_CTEST_DRIVER()
