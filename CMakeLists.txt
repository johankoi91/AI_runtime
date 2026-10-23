cmake_minimum_required(VERSION 3.20)

project(mini_llm_runtime LANGUAGES CXX)

set(HUSH_MODEL_BUNDLE_PATH "/tmp/hush-real-model.tar.gz" CACHE FILEPATH
    "Hush model bundle to embed in the macOS app")

find_package(ZLIB REQUIRED)

if(APPLE)
  enable_language(OBJCXX)
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_library(mini_runtime
  src/audio.cpp
  src/wav.cpp
  src/spectral.cpp
  src/tensor.cpp
  src/memory_pool.cpp
  src/weights.cpp
  src/onnx_weights.cpp
  src/onnx_plan.cpp
  src/hush_bundle.cpp
  src/hush_model.cpp
  src/hush_session.cpp
  src/hush_frontend.cpp
  src/onnx_executor.cpp
  src/graph.cpp
  src/executor.cpp
  src/backend/cpu_backend.cpp
  src/kernels/cpu/add.cpp
  src/kernels/cpu/mul.cpp
  src/kernels/cpu/spectrum.cpp
  src/kernels/cpu/istft.cpp
  src/kernels/cpu/activations.cpp
  src/kernels/cpu/gru.cpp
  src/kernels/cpu/conv2d.cpp
  src/kernels/cpu/conv_transpose2d.cpp
  src/kernels/cpu/erb.cpp
  src/kernels/cpu/shape.cpp
  src/kernels/cpu/matmul.cpp
)

if(APPLE)
  target_sources(mini_runtime PRIVATE
    src/backend/metal_backend.mm
    src/telemetry/influxdb_sink.mm
  )
endif()

target_include_directories(mini_runtime PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_compile_features(mini_runtime PUBLIC cxx_std_20)
target_link_libraries(mini_runtime PRIVATE ZLIB::ZLIB)

if(APPLE)
  target_link_libraries(mini_runtime PRIVATE "-framework Metal" "-framework Foundation")
endif()

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

add_executable(hush_real_probe apps/hush_real_probe.cpp)
target_link_libraries(hush_real_probe PRIVATE mini_runtime)
set_target_properties(hush_real_probe PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin
)

add_executable(hush_benchmark apps/hush_benchmark.cpp)
target_link_libraries(hush_benchmark PRIVATE mini_runtime)
set_target_properties(hush_benchmark PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin
)

add_executable(hush_stability apps/hush_stability.cpp)
target_link_libraries(hush_stability PRIVATE mini_runtime)
set_target_properties(hush_stability PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin
)

add_executable(hush_wav_demo apps/hush_wav_demo.cpp)
target_link_libraries(hush_wav_demo PRIVATE mini_runtime)
set_target_properties(hush_wav_demo PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin)

add_executable(hush_quality apps/hush_quality.cpp)
target_link_libraries(hush_quality PRIVATE mini_runtime)
set_target_properties(hush_quality PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin)

add_executable(hush_inspect apps/hush_inspect.cpp)
target_link_libraries(hush_inspect PRIVATE mini_runtime)
set_target_properties(hush_inspect PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin
)

if(APPLE)
  add_executable(mini_runtime_metal_demo apps/metal_demo.cpp)
  target_link_libraries(mini_runtime_metal_demo PRIVATE mini_runtime)
  set_target_properties(mini_runtime_metal_demo PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin
  )
  add_executable(hush_mic_demo apps/hush_mic_demo.mm)
  target_link_libraries(hush_mic_demo PRIVATE mini_runtime "-framework AVFoundation")
  set_target_properties(hush_mic_demo PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin)
  add_executable(hush_mic_ui apps/hush_mic_ui.mm)
  target_link_libraries(hush_mic_ui PRIVATE mini_runtime "-framework AVFoundation" "-framework AppKit")
  set_target_properties(hush_mic_ui PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin
    MACOSX_BUNDLE TRUE
    MACOSX_BUNDLE_INFO_PLIST ${PROJECT_SOURCE_DIR}/apps/HushMicUI-Info.plist)
  if(EXISTS "${HUSH_MODEL_BUNDLE_PATH}")
    add_custom_command(TARGET hush_mic_ui POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E make_directory
        "$<TARGET_BUNDLE_DIR:hush_mic_ui>/Contents/Resources"
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${HUSH_MODEL_BUNDLE_PATH}"
        "$<TARGET_BUNDLE_DIR:hush_mic_ui>/Contents/Resources/hush-real-model.tar.gz")
  else()
    message(WARNING "HUSH_MODEL_BUNDLE_PATH does not exist; app will require HUSH_MODEL_BUNDLE at runtime")
  endif()
endif()

include(CTest)

if(BUILD_TESTING)
  add_executable(mini_runtime_tests tests/runtime_tests.cpp)
  target_link_libraries(mini_runtime_tests PRIVATE mini_runtime)
  set_target_properties(mini_runtime_tests PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin
  )
  add_test(NAME mini_runtime_tests COMMAND ${PROJECT_BINARY_DIR}/bin/mini_runtime_tests)
endif()
