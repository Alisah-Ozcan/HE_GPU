# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
# Developer: Alişah Özcan

# Source files in this directory
set(HE_SOURCE_FILES ${HE_SOURCE_FILES}
    ${CMAKE_CURRENT_LIST_DIR}/nttparameters.cu
    ${CMAKE_CURRENT_LIST_DIR}/ntt.cu
    ${CMAKE_CURRENT_LIST_DIR}/encoding.cu
    ${CMAKE_CURRENT_LIST_DIR}/encryption.cu
    ${CMAKE_CURRENT_LIST_DIR}/decryption.cu
    ${CMAKE_CURRENT_LIST_DIR}/addition.cu
    ${CMAKE_CURRENT_LIST_DIR}/multiplication.cu
    ${CMAKE_CURRENT_LIST_DIR}/switchkey.cu
    PARENT_SCOPE
)

#set(HE_SOURCE_FILES ${HE_SOURCE_FILES} PARENT_SCOPE)