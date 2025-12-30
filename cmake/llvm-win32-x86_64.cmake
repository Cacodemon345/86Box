#
# 86Box    A hypervisor and IBM PC system emulator that specializes in
#          running old operating systems and software designed for IBM
#          PC systems and compatibles from 1981 through fairly recent
#          system designs based on the PCI bus.
#
#          This file is part of the 86Box distribution.
#
#          CMake toolchain file for Clang on Windows builds (x64/AMD64 target).
#
# Authors: David Hrdlička, <hrdlickadavid@outlook.com>
#
#          Copyright 2021 David Hrdlička.
#

include(${CMAKE_CURRENT_LIST_DIR}/flags-gcc-x86_64.cmake)

# Use the GCC-compatible Clang executables in order to use our flags
set(CMAKE_C_COMPILER    clang)
set(CMAKE_CXX_COMPILER  clang++)

# Use llvm-rc for resource compilation in CLANG64 environment
set(CMAKE_RC_COMPILER   llvm-rc)

set(CMAKE_C_COMPILER_TARGET     x86_64-w64-windows-gnu)
set(CMAKE_CXX_COMPILER_TARGET   x86_64-w64-windows-gnu)

set(CMAKE_SYSTEM_PROCESSOR AMD64)

# Override debug flags to ensure PDB generation
set(CMAKE_C_FLAGS_DEBUG "-g -gcodeview -O0" CACHE STRING "Flags used by the C compiler during DEBUG builds." FORCE)
set(CMAKE_CXX_FLAGS_DEBUG "-g -gcodeview -O0" CACHE STRING "Flags used by the CXX compiler during DEBUG builds." FORCE)
set(CMAKE_EXE_LINKER_FLAGS_DEBUG "-fuse-ld=lld -Wl,--pdb=" CACHE STRING "Flags used by the linker during DEBUG builds." FORCE)

# TODO: set the vcpkg target triplet perhaps?
