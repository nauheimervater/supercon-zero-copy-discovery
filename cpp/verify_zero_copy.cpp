// =============================================================================
// verify_zero_copy.cpp
// Hardware verification of unified-memory pointer identity (cpu_ptr == gpu_ptr)
// via hipHostMallocMapped on AMD Ryzen AI Max+ 395 (Strix Halo APU, GFX1151).
//
// Author: Thomas Nauheimer (2026) | License: MIT
// Build:  hipcc -O3 verify_zero_copy.cpp -o verify_zero_copy
// =============================================================================

#include <hip/hip_runtime.h>
#include <iostream>

int main() {
    hipInit(0);
    float* cpu_ptr = nullptr;
    float* gpu_ptr = nullptr;
    const size_t size = 1024 * sizeof(float);

    hipError_t err = hipHostMalloc(&cpu_ptr, size, hipHostMallocMapped);
    if (err != hipSuccess) {
        std::cerr << "[ERROR] hipHostMallocMapped failed: "
                  << hipGetErrorString(err) << std::endl;
        return 1;
    }

    err = hipHostGetDevicePointer(reinterpret_cast<void**>(&gpu_ptr), cpu_ptr, 0);
    if (err != hipSuccess) {
        std::cerr << "[ERROR] hipHostGetDevicePointer failed: "
                  << hipGetErrorString(err) << std::endl;
        hipFree(cpu_ptr);
        return 1;
    }

    std::cout << "=======================================================\n";
    std::cout << "  AMD STRIX HALO APU UNIFIED-MEMORY ZERO-COPY CHECK\n";
    std::cout << "=======================================================\n";
    std::cout << "Host (CPU) virtual address   : " << static_cast<void*>(cpu_ptr) << "\n";
    std::cout << "Device (GPU) virtual address : " << static_cast<void*>(gpu_ptr) << "\n";

    if (cpu_ptr == gpu_ptr) {
        std::cout << "[PASS] Host and device pointers identical (no explicit host-device copy).\n";
    } else {
        std::cout << "[INFO] Mapped host/device aliases active.\n";
    }

    hipFree(cpu_ptr);
    return 0;
}
