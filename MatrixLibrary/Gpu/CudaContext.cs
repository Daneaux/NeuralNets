using System.Runtime.InteropServices;

namespace MatrixLibrary.Gpu
{
    /// <summary>
    /// Singleton managing CUDA device initialization and CUBLAS handle lifecycle.
    /// Thread-safe via Lazy&lt;T&gt;.
    /// </summary>
    internal sealed class CudaContext : IDisposable
    {
        private static readonly Lazy<CudaContext> _instance =
            new Lazy<CudaContext>(() => new CudaContext());

        public static CudaContext Instance => _instance.Value;

        public IntPtr CublasHandle { get; private set; }
        public bool IsInitialized { get; private set; }

        private CudaContext()
        {
            Initialize();
        }

        private void Initialize()
        {
            var setDeviceStatus = CublasNativeMethods.cudaSetDevice(0);
            CheckCudaStatus(setDeviceStatus, "cudaSetDevice");

            var createStatus = CublasNativeMethods.cublasCreate_v2(out var handle);
            CheckCublasStatus(createStatus, "cublasCreate_v2");

            CublasHandle = handle;
            IsInitialized = true;

            AppDomain.CurrentDomain.ProcessExit += (_, _) => Dispose();
        }

        /// <summary>
        /// Checks if a CUDA-capable GPU is available without fully initializing.
        /// Catches DllNotFoundException for systems without CUDA installed.
        /// </summary>
        public static bool IsGpuAvailable()
        {
            try
            {
                var status = CublasNativeMethods.cudaGetDeviceCount(out int count);
                return status == CudaError.Success && count > 0;
            }
            catch (DllNotFoundException)
            {
                return false;
            }
            catch (EntryPointNotFoundException)
            {
                return false;
            }
        }

        internal static void CheckCudaStatus(CudaError status, string functionName)
        {
            if (status != CudaError.Success)
            {
                string errorMsg;
                try
                {
                    IntPtr errorPtr = CublasNativeMethods.cudaGetErrorString(status);
                    errorMsg = Marshal.PtrToStringAnsi(errorPtr) ?? status.ToString();
                }
                catch
                {
                    errorMsg = status.ToString();
                }
                throw new CudaException($"{functionName} failed with error: {errorMsg} ({(int)status})");
            }
        }

        internal static void CheckCublasStatus(CublasStatus status, string functionName)
        {
            if (status != CublasStatus.Success)
            {
                throw new CudaException($"{functionName} failed with CUBLAS status: {status} ({(int)status})");
            }
        }

        public void Dispose()
        {
            if (IsInitialized)
            {
                CublasNativeMethods.cublasDestroy_v2(CublasHandle);
                CublasHandle = IntPtr.Zero;
                IsInitialized = false;
            }
        }
    }

    /// <summary>
    /// Exception thrown for CUDA/CUBLAS runtime errors.
    /// </summary>
    public class CudaException : Exception
    {
        public CudaException(string message) : base(message) { }
    }
}
