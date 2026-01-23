using System.Runtime.InteropServices;

namespace MatrixLibrary.Gpu
{
    /// <summary>
    /// P/Invoke declarations for CUDA Runtime and CUBLAS (CUDA 13.x).
    /// </summary>
    internal static class CublasNativeMethods
    {
        private const string CUBLAS_DLL = "cublas64_13.dll";
        private const string CUDART_DLL = "cudart64_13.dll";

        // ============================================================
        // CUDA Runtime API (cudart64_13.dll)
        // ============================================================

        [DllImport(CUDART_DLL, CallingConvention = CallingConvention.Cdecl)]
        internal static extern CudaError cudaGetDeviceCount(out int count);

        [DllImport(CUDART_DLL, CallingConvention = CallingConvention.Cdecl)]
        internal static extern CudaError cudaSetDevice(int device);

        [DllImport(CUDART_DLL, CallingConvention = CallingConvention.Cdecl)]
        internal static extern CudaError cudaMalloc(out IntPtr devPtr, ulong size);

        [DllImport(CUDART_DLL, CallingConvention = CallingConvention.Cdecl)]
        internal static extern CudaError cudaFree(IntPtr devPtr);

        [DllImport(CUDART_DLL, CallingConvention = CallingConvention.Cdecl)]
        internal static extern CudaError cudaMemcpy(IntPtr dst, IntPtr src, ulong count, CudaMemcpyKind kind);

        [DllImport(CUDART_DLL, CallingConvention = CallingConvention.Cdecl)]
        internal static extern CudaError cudaDeviceSynchronize();

        [DllImport(CUDART_DLL, CallingConvention = CallingConvention.Cdecl)]
        internal static extern IntPtr cudaGetErrorString(CudaError error);

        // ============================================================
        // CUBLAS API (cublas64_12.dll)
        // ============================================================

        [DllImport(CUBLAS_DLL, CallingConvention = CallingConvention.Cdecl)]
        internal static extern CublasStatus cublasCreate_v2(out IntPtr handle);

        [DllImport(CUBLAS_DLL, CallingConvention = CallingConvention.Cdecl)]
        internal static extern CublasStatus cublasDestroy_v2(IntPtr handle);

        [DllImport(CUBLAS_DLL, CallingConvention = CallingConvention.Cdecl)]
        internal static extern unsafe CublasStatus cublasSgemm_v2(
            IntPtr handle,
            CublasOperation transa,
            CublasOperation transb,
            int m, int n, int k,
            float* alpha,
            IntPtr A, int lda,
            IntPtr B, int ldb,
            float* beta,
            IntPtr C, int ldc);

        [DllImport(CUBLAS_DLL, CallingConvention = CallingConvention.Cdecl)]
        internal static extern unsafe CublasStatus cublasSgemv_v2(
            IntPtr handle,
            CublasOperation trans,
            int m, int n,
            float* alpha,
            IntPtr A, int lda,
            IntPtr x, int incx,
            float* beta,
            IntPtr y, int incy);

        [DllImport(CUBLAS_DLL, CallingConvention = CallingConvention.Cdecl)]
        internal static extern unsafe CublasStatus cublasSaxpy_v2(
            IntPtr handle,
            int n,
            float* alpha,
            IntPtr x, int incx,
            IntPtr y, int incy);

        [DllImport(CUBLAS_DLL, CallingConvention = CallingConvention.Cdecl)]
        internal static extern unsafe CublasStatus cublasSscal_v2(
            IntPtr handle,
            int n,
            float* alpha,
            IntPtr x, int incx);

        [DllImport(CUBLAS_DLL, CallingConvention = CallingConvention.Cdecl)]
        internal static extern unsafe CublasStatus cublasSger_v2(
            IntPtr handle,
            int m, int n,
            float* alpha,
            IntPtr x, int incx,
            IntPtr y, int incy,
            IntPtr A, int lda);

        [DllImport(CUBLAS_DLL, CallingConvention = CallingConvention.Cdecl)]
        internal static extern unsafe CublasStatus cublasSgeam(
            IntPtr handle,
            CublasOperation transa,
            CublasOperation transb,
            int m, int n,
            float* alpha,
            IntPtr A, int lda,
            float* beta,
            IntPtr B, int ldb,
            IntPtr C, int ldc);

        [DllImport(CUBLAS_DLL, CallingConvention = CallingConvention.Cdecl)]
        internal static extern unsafe CublasStatus cublasScopy_v2(
            IntPtr handle,
            int n,
            IntPtr x, int incx,
            IntPtr y, int incy);
    }

    // ============================================================
    // Enums
    // ============================================================

    internal enum CudaMemcpyKind
    {
        HostToHost = 0,
        HostToDevice = 1,
        DeviceToHost = 2,
        DeviceToDevice = 3
    }

    internal enum CudaError
    {
        Success = 0,
        InvalidValue = 1,
        MemoryAllocation = 2,
        InitializationError = 3,
        // Add more as needed
    }

    internal enum CublasStatus
    {
        Success = 0,
        NotInitialized = 1,
        AllocFailed = 3,
        InvalidValue = 7,
        ArchMismatch = 8,
        MappingError = 11,
        ExecutionFailed = 13,
        InternalError = 14,
        NotSupported = 15,
    }

    internal enum CublasOperation
    {
        None = 0,       // CUBLAS_OP_N
        Transpose = 1,  // CUBLAS_OP_T
        ConjugateTranspose = 2  // CUBLAS_OP_C
    }
}
