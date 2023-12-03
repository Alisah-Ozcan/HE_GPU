# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
# Developer: Alişah Özcan

add_subdirectory(gpu_kernels)

# Source files in this directory
set(HE_SOURCE_FILES ${HE_SOURCE_FILES}
    ${CMAKE_CURRENT_LIST_DIR}/common.cu
    ${CMAKE_CURRENT_LIST_DIR}/context.cu
    ${CMAKE_CURRENT_LIST_DIR}/contextpool.cu
    ${CMAKE_CURRENT_LIST_DIR}/keygeneration.cu
    ${CMAKE_CURRENT_LIST_DIR}/encoder.cu
    ${CMAKE_CURRENT_LIST_DIR}/encryptor.cu
    ${CMAKE_CURRENT_LIST_DIR}/decryptor.cu
    ${CMAKE_CURRENT_LIST_DIR}/operator.cu
    PARENT_SCOPE
)

#set(HE_SOURCE_FILES ${HE_SOURCE_FILES} PARENT_SCOPE)