namespace MatrixLibrary
{
    /// <summary>
    /// Defines the available backend implementations for matrix and vector operations.
    /// </summary>
    public enum MatrixBackend
    {
        /// <summary>
        /// Pure C# implementation without SIMD optimizations.
        /// Suitable for small matrices and compatibility fallback.
        /// </summary>
        Software,

        /// <summary>
        /// AVX-512 SIMD implementation for CPU.
        /// Recommended for medium to large matrices on x86-64 processors.
        /// </summary>
        AVX,

        /// <summary>
        /// CUDA/OpenCL implementation for GPU acceleration.
        /// Recommended for very large matrices and batch operations.
        /// </summary>
        GPU
    }
}
