using System;
using System.Runtime.Intrinsics.X86;

namespace MatrixLibrary
{
    /// <summary>
    /// Provides backend capability detection and recommendation.
    /// </summary>
    public static class BackendSelector
    {
        private static bool? _avxAvailable;
        private static bool? _gpuAvailable;

        /// <summary>
        /// Detects if AVX-512 is available on the current system.
        /// </summary>
        /// <returns>True if AVX-512 is available</returns>
        public static bool IsAVXAvailable()
        {
            if (_avxAvailable.HasValue)
                return _avxAvailable.Value;

            _avxAvailable = Avx512F.IsSupported;
            return _avxAvailable.Value;
        }

        /// <summary>
        /// Detects if GPU acceleration is available on the current system.
        /// </summary>
        /// <returns>True if GPU is available</returns>
        public static bool IsGPUAvailable()
        {
            if (_gpuAvailable.HasValue)
                return _gpuAvailable.Value;

            try
            {
                _gpuAvailable = Gpu.CudaContext.IsGpuAvailable();
            }
            catch
            {
                _gpuAvailable = false;
            }
            return _gpuAvailable.Value;
        }

        /// <summary>
        /// Gets the recommended backend based on system capabilities.
        /// </summary>
        /// <returns>Recommended backend</returns>
        public static MatrixBackend GetRecommendedBackend()
        {
            // Check GPU first (best performance for large matrices)
            if (IsGPUAvailable())
                return MatrixBackend.GPU;

            // Check AVX next (good performance for medium matrices)
            if (IsAVXAvailable())
                return MatrixBackend.AVX;

            // Fall back to Software (compatible with all systems)
            return MatrixBackend.Software;
        }

        /// <summary>
        /// Validates if a specific backend is available on the current system.
        /// </summary>
        /// <param name="backend">Backend to check</param>
        /// <returns>True if backend is available</returns>
        public static bool IsBackendAvailable(MatrixBackend backend)
        {
            return backend switch
            {
                MatrixBackend.Software => true, // Always available
                MatrixBackend.AVX => IsAVXAvailable(),
                MatrixBackend.GPU => IsGPUAvailable(),
                _ => false
            };
        }

        /// <summary>
        /// Gets a description of the recommended backend and why it was chosen.
        /// </summary>
        /// <returns>Description string</returns>
        public static string GetRecommendedBackendDescription()
        {
            var recommended = GetRecommendedBackend();
            return recommended switch
            {
                MatrixBackend.GPU => "GPU (CUDA/OpenCL) - Recommended for large matrices and batch operations",
                MatrixBackend.AVX => "AVX-512 SIMD - Recommended for medium to large matrices on x86-64 processors",
                MatrixBackend.Software => "Software (Pure C#) - Fallback for compatibility and small matrices",
                _ => "Unknown backend"
            };
        }

        /// <summary>
        /// Gets the recommended backend based on matrix size.
        /// </summary>
        /// <param name="totalElements">Total number of elements in the matrix</param>
        /// <returns>Recommended backend for the given size</returns>
        public static MatrixBackend GetRecommendedBackendForSize(int totalElements)
        {
            // Heuristics for backend selection based on matrix size
            if (totalElements < 100)
                return MatrixBackend.Software; // Overhead too high for AVX/GPU

            if (totalElements < 10000)
                return IsAVXAvailable() ? MatrixBackend.AVX : MatrixBackend.Software;

            return IsGPUAvailable() ? MatrixBackend.GPU :
                   IsAVXAvailable() ? MatrixBackend.AVX :
                   MatrixBackend.Software;
        }
    }
}
