idf_component_register(SRCS "main.cpp" INCLUDE_DIRS ".")

# Modelo quantizado pelo ESP-PPQ (../../quantizar_espdl.py), embutido em rodata e lido
# por dl::Model com fbs::MODEL_LOCATION_IN_FLASH_RODATA (padrão dos exemplos do ESP-DL).
set(ARQUIVO_MODELO ${CMAKE_CURRENT_SOURCE_DIR}/../../modelos/classificador.espdl)
idf_build_get_property(component_targets __COMPONENT_TARGETS)
if("___idf_espressif__esp-dl" IN_LIST component_targets)
  idf_component_get_property(espdl_dir espressif__esp-dl COMPONENT_DIR)
elseif("___idf_esp-dl" IN_LIST component_targets)
  idf_component_get_property(espdl_dir esp-dl COMPONENT_DIR)
endif()
include(${espdl_dir}/fbs_loader/cmake/utilities.cmake)
target_add_aligned_binary_data(${COMPONENT_LIB} ${ARQUIVO_MODELO} BINARY)
