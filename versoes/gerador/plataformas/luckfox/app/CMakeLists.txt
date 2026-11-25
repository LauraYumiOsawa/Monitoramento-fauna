# Armadilha da versão @@ID@@ (Luckfox Pico RV1106G3). Compilar com ../compilar.sh.
cmake_minimum_required(VERSION 3.10)
project(armadilha_@@ID@@ CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# opencv-mobile para Luckfox Pico: VideoCapture já carrega o rkaiq (ISP) da placa.
find_package(OpenCV REQUIRED)
if(NOT RKNN_API_DIR)
  message(FATAL_ERROR "Informe -DRKNN_API_DIR=<rknn-toolkit2>/rknpu2/runtime/Linux/librknn_api")
endif()

add_executable(armadilha src/main.cc)
target_include_directories(armadilha PRIVATE ${RKNN_API_DIR}/include)
target_link_libraries(armadilha ${OpenCV_LIBS} ${RKNN_API_DIR}/armhf-uclibc/librknnmrt.so)
install(TARGETS armadilha DESTINATION .)
