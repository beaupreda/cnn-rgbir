#
#   Copyright 2013 Pixar
#
#   Licensed under the Apache License, Version 2.0 (the "Apache License")
#   with the following modification; you may not use this file except in
#   compliance with the Apache License and the following modification to it:
#   Section 6. Trademarks. is deleted and replaced with:
#
#   6. Trademarks. This License does not grant permission to use the trade
#      names, trademarks, service marks, or product names of the Licensor
#      and its affiliates, except as required to comply with Section 4(c) of
#      the License and to reproduce the content of the NOTICE file.
#
#   You may obtain a copy of the Apache License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the Apache License with the above modification is
#   distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
#   KIND, either express or implied. See the Apache License for the specific
#   language governing permissions and limitations under the Apache License.
#

# Try to find GLFW library and include path.
# Once done this will define
#
# GLFW_FOUND
# GLFW_INCLUDE_DIR
# GLFW_LIBRARIES
#

set(GLFW_FOUND 0)

find_path(GLFW_INCLUDE_DIR
    NAMES
        GLFW/glfw3.h
    HINTS
        "${GLFW_LOCATION}/include"
        "$ENV{GLFW_LOCATION}/include"
        "$ENV{USER_DEVELOP}/glfw/include"
        "$ENV{USER_DEVELOP}/vendor/glfw/include"
    PATHS
        "$ENV{PROGRAMFILES}/GLFW/include"
        "${OPENGL_INCLUDE_DIR}"
        /usr/openwin/share/include
        /usr/openwin/include
        /usr/X11R6/include
        /usr/include/X11
        /opt/graphics/OpenGL/include
        /opt/graphics/OpenGL/contrib/libglfw
        /usr/local/include
        /usr/include/GL
        /usr/include
    DOC
        "The directory where GLFW/glfw3.h resides"
)

find_path(GLFW_INCLUDE_DIR
    NAMES
        GL/glfw.h
    HINTS
        "${GLFW_LOCATION}/include"
        "$ENV{GLFW_LOCATION}/include"
        "$ENV{USER_DEVELOP}/glfw/include"
        "$ENV{USER_DEVELOP}/vendor/glfw/include"
    PATHS
        "$ENV{PROGRAMFILES}/GLFW/include"
        "${OPENGL_INCLUDE_DIR}"
        /usr/openwin/share/include
        /usr/openwin/include
        /usr/X11R6/include
        /usr/include/X11
        /opt/graphics/OpenGL/include
        /opt/graphics/OpenGL/contrib/libglfw
        /usr/local/include
        /usr/include/GL
        /usr/include
    DOC
        "The directory where GL/glfw.h resides"
)

if(WIN32)
    if(CYGWIN)
        find_library(GLFW_glfw_LIBRARY
            NAMES
                glfw32
            HINTS
                "${GLFW_LOCATION}/lib"
                "${GLFW_LOCATION}/lib/x64"
                "$ENV{GLFW_LOCATION}/lib"
                "$ENV{USER_DEVELOP}/glfw/build/src/Release/"
                "$ENV{USER_DEVELOP}/vendor/glfw/build/src/Release/"
            PATHS
                "${OPENGL_LIBRARY_DIR}"
                /usr/lib
                /usr/lib/w32api
                /usr/local/lib
                /usr/X11R6/lib
            DOC
                "The GLFW library"
        )
    else()
        find_library(GLFW_glfw_LIBRARY
            NAMES
                glfw32
                glfw32s
                glfw
                glfw3
                glfw3dll
            HINTS
                "${GLFW_LOCATION}/lib"
                "${GLFW_LOCATION}/lib/x64"
                "${GLFW_LOCATION}/lib-msvc110"
                "${GLFW_LOCATION}/lib-vc2012"
                "$ENV{GLFW_LOCATION}/lib"
                "$ENV{GLFW_LOCATION}/lib/x64"
                "$ENV{GLFW_LOCATION}/lib-msvc110"
                "$ENV{GLFW_LOCATION}/lib-vc2012"
                "$ENV{USER_DEVELOP}/glfw/build/src/Release/"
                "$ENV{USER_DEVELOP}/vendor/glfw/build/src/Release/"
            PATHS
                "$ENV{PROGRAMFILES}/GLFW/lib"
                "${OPENGL_LIBRARY_DIR}"
            DOC
                "The GLFW library"
        )
    endif()
    set(GLFW_FOUND_DEPS 1)
else()
    if(APPLE)
        find_library(GLFW_glfw_LIBRARY glfw
            NAMES
                glfw
                glfw3
            HINTS
                "${GLFW_LOCATION}/lib"
                "${GLFW_LOCATION}/lib/cocoa"
                "$ENV{GLFW_LOCATION}/lib"
                "$ENV{GLFW_LOCATION}/lib/cocoa"
            PATHS
                /usr/local/lib
        )
        set(GLFW_cocoa_LIBRARY "-framework Cocoa" CACHE STRING "Cocoa framework for OSX")
        set(GLFW_corevideo_LIBRARY "-framework CoreVideo" CACHE STRING "CoreVideo framework for OSX")
        set(GLFW_iokit_LIBRARY "-framework IOKit" CACHE STRING "IOKit framework for OSX")
        set(GLFW_FOUND_DEPS 1)
    else() # (*)NIX
        if(GLFW_FIND_REQUIRED)
            find_package(Threads REQUIRED)
            find_package(X11 REQUIRED)
            if(NOT X11_Xrandr_FOUND)
                message(FATAL_ERROR "Xrandr library not found - required for GLFW")
            endif()
            if(NOT X11_xf86vmode_FOUND)
                message(FATAL_ERROR "xf86vmode library not found - required for GLFW")
            endif()
            if(NOT X11_Xcursor_FOUND)
                message(FATAL_ERROR "Xcursor library not found - required for GLFW")
            endif()
            if(NOT X11_Xinerama_FOUND)
                message(FATAL_ERROR "Xinerama library not found - required for GLFW")
            endif()
            set(GLFW_FOUND_DEPS 1)
        else()
            find_package(Threads)
            if(NOT Threads_FOUND)
                if(NOT GLFW_FIND_QUIETLY)
                    message(STATUS "Threading library not found - required for GLFW")
                endif()
            else()
                find_package(X11)
                if(NOT X11_FOUND)
                    if(NOT GLFW_FIND_QUIETLY)
                        message(STATUS "X11 library not found - required for GLFW")
                    endif()
                elseif(NOT X11_Xrandr_FOUND)
                    if(NOT GLFW_FIND_QUIETLY)
                        message(STATUS "Xrandr library not found - required for GLFW")
                    endif()
                elseif(NOT X11_xf86vmode_FOUND)
                    if(NOT GLFW_FIND_QUIETLY)
                        message(STATUS "xf86vmode library not found - required for GLFW")
                    endif()
                elseif(NOT X11_Xcursor_FOUND)
                    if(NOT GLFW_FIND_QUIETLY)
                        message(STATUS "Xcursor library not found - required for GLFW")
                    endif()
                elseif(NOT X11_Xinerama_FOUND)
                    if(NOT GLFW_FIND_QUIETLY)
                        message(STATUS "Xinerama library not found - required for GLFW")
                    endif()
                else()
                    set(GLFW_FOUND_DEPS 1)
                endif()
            endif()
        endif()

        list(APPEND GLFW_x11_LIBRARY "${X11_Xrandr_LIB}" "${X11_Xxf86vm_LIB}" "${X11_Xcursor_LIB}" "${X11_Xinerama_LIB}" "${CMAKE_THREAD_LIBS_INIT}" -lrt -lXi)
        find_library(GLFW_glfw_LIBRARY
            NAMES
                glfw
                glfw3
            HINTS
                "${GLFW_LOCATION}/lib"
                "$ENV{GLFW_LOCATION}/lib"
                "${GLFW_LOCATION}/lib/x11"
                "$ENV{GLFW_LOCATION}/lib/x11"
            PATHS
                /usr/lib64
                /usr/lib
                /usr/lib/${CMAKE_LIBRARY_ARCHITECTURE}
                /usr/local/lib64
                /usr/local/lib
                /usr/local/lib/${CMAKE_LIBRARY_ARCHITECTURE}
                /usr/openwin/lib
                /usr/X11R6/lib
            DOC
                "The GLFW library"
        )
    endif()
endif()

if(GLFW_INCLUDE_DIR)
    if(GLFW_glfw_LIBRARY)
        set(GLFW_LIBRARIES
            "${GLFW_glfw_LIBRARY}"
            "${GLFW_x11_LIBRARY}"
            "${GLFW_cocoa_LIBRARY}"
            "${GLFW_iokit_LIBRARY}"
            "${GLFW_corevideo_LIBRARY}"
        )
        set(GLFW_FOUND 1)
        set(GLFW_LIBRARY "${GLFW_LIBRARIES}")
        set(GLFW_INCLUDE_PATH "${GLFW_INCLUDE_DIR}")
    endif(GLFW_glfw_LIBRARY)
    # Tease the GLFW_VERSION numbers from the lib headers
    function(parseVersion FILENAME VARNAME)
        set(PATTERN "^#define ${VARNAME}.*$")
        file(STRINGS "${GLFW_INCLUDE_DIR}/${FILENAME}" TMP REGEX ${PATTERN})
        string(REGEX MATCHALL "[0-9]+" TMP ${TMP})
        set(${VARNAME} ${TMP} PARENT_SCOPE)
    endfunction()
    if(EXISTS "${GLFW_INCLUDE_DIR}/GL/glfw.h")
        parseVersion(GL/glfw.h GLFW_VERSION_MAJOR)
        parseVersion(GL/glfw.h GLFW_VERSION_MINOR)
        parseVersion(GL/glfw.h GLFW_VERSION_REVISION)
    elseif(EXISTS "${GLFW_INCLUDE_DIR}/GLFW/glfw3.h")
        parseVersion(GLFW/glfw3.h GLFW_VERSION_MAJOR)
        parseVersion(GLFW/glfw3.h GLFW_VERSION_MINOR)
        parseVersion(GLFW/glfw3.h GLFW_VERSION_REVISION)
    endif()
    if(${GLFW_VERSION_MAJOR} OR ${GLFW_VERSION_MINOR} OR ${GLFW_VERSION_REVISION})
        set(GLFW_VERSION "${GLFW_VERSION_MAJOR}.${GLFW_VERSION_MINOR}.${GLFW_VERSION_REVISION}")
        set(GLFW_VERSION_STRING "${GLFW_VERSION}")
        mark_as_advanced(GLFW_VERSION)
    endif()
endif(GLFW_INCLUDE_DIR)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(GLFW
    REQUIRED_VARS
        GLFW_INCLUDE_DIR
        GLFW_LIBRARIES
        GLFW_FOUND_DEPS
    VERSION_VAR
        GLFW_VERSION
)
if(GLFW_FOUND)
    mark_as_advanced(
        GLFW_INCLUDE_DIR
        GLFW_LIBRARIES
        GLFW_glfw_LIBRARY
        GLFW_cocoa_LIBRARY
    )
endif()
