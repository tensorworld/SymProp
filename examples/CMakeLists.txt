add_executable(symtens_ttmc symtens_ttmc.cpp)

target_link_libraries(symtens_ttmc PRIVATE ${PROJECT_NAME})

add_executable(symtucker symtucker.cpp)

target_link_libraries(symtucker PRIVATE ${PROJECT_NAME})

add_executable(s3ttmctc s3ttmctc.cpp)

target_link_libraries(s3ttmctc PRIVATE ${PROJECT_NAME})