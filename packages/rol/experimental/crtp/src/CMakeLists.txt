SET(HEADERS "")
SET(SOURCES "")

INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR})

SET(SRC ${CMAKE_CURRENT_SOURCE_DIR})
SET_AND_INC_DIRS(DIR ${SRC})
INCLUDE_DIRECTORIES(${DIR})
APPEND_GLOB(HEADERS ${DIR}/XROL_Core.hpp)

SET_AND_INC_DIRS(DIR ${SRC}/utility)
INCLUDE_DIRECTORIES(${DIR})
APPEND_GLOB(HEADERS ${DIR}/*.hpp)

SET_AND_INC_DIRS(DIR ${SRC}/vector)
INCLUDE_DIRECTORIES(${DIR})
APPEND_GLOB(HEADERS ${DIR}/*.hpp)

SET_AND_INC_DIRS(DIR ${SRC}/function)
INCLUDE_DIRECTORIES(${DIR})
APPEND_GLOB(HEADERS ${DIR}/*.hpp)


TRIBITS_ADD_LIBRARY(
  xrol
  HEADERS ${HEADERS}
)
