# function to add tests
set(CATCH_DIR "${CMAKE_SOURCE_DIR}/extern/catch")

function(add_test_executable name)
  add_executable(${name} ${ARGN} utils.cpp)
  target_link_libraries(${name} PRIVATE Catch2::Catch2WithMain ${PROJECT_NAME})
  target_link_libraries(${name} PRIVATE ${MATH_LIBS})
endfunction()


add_test_executable(test_symtensor test_symtensor.cpp)
add_test_executable(test_dtree test_dtree.cpp)
add_test_executable(test_compressed_dtree test_compressed_dtree.cpp)
add_test_executable(test_matrixops test_matrixops.cpp)



