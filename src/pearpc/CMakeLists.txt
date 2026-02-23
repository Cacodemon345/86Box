# It'd be cool if PowerPC 615 ever became a real thing, but alas...

add_library(pearpc OBJECT
    uaeglue.cpp
    cpu/cpu_generic/ppc_alu.cpp
    cpu/cpu_generic/ppc_cpu.cpp
    cpu/cpu_generic/ppc_dec.cpp
    cpu/cpu_generic/ppc_exc.cpp
    cpu/cpu_generic/ppc_fpu.cpp
    cpu/cpu_generic/ppc_mmu.cpp
    cpu/cpu_generic/ppc_opc.cpp
    cpu/cpu_generic/ppc_vec.cpp
)
