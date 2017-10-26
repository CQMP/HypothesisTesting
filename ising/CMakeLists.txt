cmake_minimum_required(VERSION 3.1)

project (ising)

# see https://blog.kitware.com/upcoming-in-cmake-2-8-12-osx-rpath-support/
set(CMAKE_MACOSX_RPATH 1)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH true)

# set (CMAKE_CXX_STANDARD 11)

find_package(FFTW3 REQUIRED)
find_package(ALPSCore REQUIRED)

include_directories(${FFTW3_INCLUDE_DIRS})

add_executable(ising-bin
               main.cc)
set_target_properties(ising-bin
                      PROPERTIES OUTPUT_NAME ising)
target_link_libraries(ising-bin
                      ${ALPSCore_LIBRARIES}
                      ${FFTW3_LIBRARIES}
                      )

if(NOT CMAKE_BUILD_TYPE)
    message(WARNING "Preparing DEFAULT build (you want Release or Debug)...")
    add_definitions(-DALPS_GF_DEBUG)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address,undefined")
elseif(CMAKE_BUILD_TYPE MATCHES ^Release|RelWithDebInfo$)
    message("Preparing RELEASE build ...")
    add_definitions(-DBOOST_DISABLE_ASSERTS)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ffast-math -flto")
elseif(CMAKE_BUILD_TYPE STREQUAL Debug)
    message("Preparing DEBUG build ...")
    add_definitions(-DALPS_GF_DEBUG)
else()
    message(FATAL_ERROR "Unknown build type: ${CMAKE_BUILD_TYPE}" )
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")

# See <http://stackoverflow.com/questions/16074598>
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set (CMAKE_INSTALL_PREFIX "$ENV{HOME}/.local" CACHE PATH
         "Default install path" FORCE)
endif()

install(DIRECTORY .
        DESTINATION include/ising
        FILES_MATCHING PATTERN "*.hh"
        )
install(TARGETS ising-bin
        RUNTIME DESTINATION "bin/"
        LIBRARY DESTINATION "lib/"
        )
