file(GLOB refactor_sources ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
file(GLOB refactor_headers ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp)
file(GLOB refactor_tests ${CMAKE_CURRENT_SOURCE_DIR}/*_test.cpp)

add_library(refactor ${refactor_headers} ${refactor_sources})
target_link_libraries(refactor PUBLIC serac_functional)
target_include_directories(refactor PUBLIC ${CMAKE_SOURCE_DIR})

foreach(filename ${refactor_tests})
  message("${filename}")
  get_filename_component(testname ${filename} NAME_WE)
  add_executable(${testname} ${filename})
  target_link_libraries(${testname} PUBLIC refactor)
endforeach(filename ${refactor_tests})
