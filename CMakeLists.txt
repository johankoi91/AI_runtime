cmake_minimum_required(VERSION 3.20)

project(mini_llm_runtime LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_library(mini_runtime
  src/tensor.cpp
  src/memory_pool.cpp
  src/graph.cpp
  src/executor.cpp
  src/backend/cpu_backend.cpp
  src/kernels/cpu/add.cpp
  src/kernels/cpu/matmul.cpp
)

target_include_directories(mini_runtime PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_compile_features(mini_runtime PUBLIC cxx_std_20)

if(MSVC)
  target_compile_options(mini_runtime PRIVATE /W4)
else()
  target_compile_options(mini_runtime PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_executable(mini_runtime_demo apps/demo.cpp)
target_link_libraries(mini_runtime_demo PRIVATE mini_runtime)
set_target_properties(mini_runtime_demo PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin
)

include(CTest)

if(BUILD_TESTING)
  add_executable(mini_runtime_tests tests/runtime_tests.cpp)
  target_link_libraries(mini_runtime_tests PRIVATE mini_runtime)
  set_target_properties(mini_runtime_tests PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin
  )
  add_test(NAME mini_runtime_tests COMMAND ${PROJECT_BINARY_DIR}/bin/mini_runtime_tests)
endif()
