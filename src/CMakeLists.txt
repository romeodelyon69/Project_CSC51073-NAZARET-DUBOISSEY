# Library sources
set(LIB_SOURCES
    EyeToolKit.cpp
    Nlib.cpp
)

set(LIB_HEADERS
    ../include/EyeToolKit.h
    ../include/Nlib.h
)

# Create library
add_library(eyetracking_lib STATIC ${LIB_SOURCES} ${LIB_HEADERS})

target_link_libraries(eyetracking_lib
    ${OpenCV_LIBS}
)

# Set include directory for the library
target_include_directories(eyetracking_lib PUBLIC
    ${CMAKE_SOURCE_DIR}/include
)

# Main application executable
add_executable(firstTest firstTest.cpp)
target_link_libraries(firstTest
    eyetracking_lib
    ${OpenCV_LIBS}
)



# Simple test executable
add_executable(test test.cpp)
target_link_libraries(test
    eyetracking_lib
    ${OpenCV_LIBS}
)

# Copy model files to build directory if they exist
if(EXISTS "${CMAKE_SOURCE_DIR}/eye_mlp_model.pth")
    file(COPY "${CMAKE_SOURCE_DIR}/eye_mlp_model.pth" 
         DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
endif()

