namespace MatrixLibrary.Gpu
{
    /// <summary>
    /// Static helpers for GPU memory allocation, deallocation, and host-device transfers.
    /// All methods use 'fixed' to pin managed arrays for cudaMemcpy.
    /// </summary>
    internal static class GpuMemoryManager
    {
        /// <summary>
        /// Allocates device memory for the specified number of floats.
        /// </summary>
        internal static IntPtr Allocate(int floatCount)
        {
            ulong sizeInBytes = (ulong)(floatCount * sizeof(float));
            var status = CublasNativeMethods.cudaMalloc(out IntPtr devicePtr, sizeInBytes);
            CudaContext.CheckCudaStatus(status, "cudaMalloc");
            return devicePtr;
        }

        /// <summary>
        /// Frees device memory. Safe to call with IntPtr.Zero.
        /// </summary>
        internal static void Free(IntPtr devicePtr)
        {
            if (devicePtr != IntPtr.Zero)
            {
                CublasNativeMethods.cudaFree(devicePtr);
            }
        }

        /// <summary>
        /// Copies a 2D float array (row-major, contiguous) from host to device.
        /// </summary>
        internal static unsafe void CopyToDevice(float[,] hostData, IntPtr devicePtr, int rows, int cols)
        {
            ulong sizeInBytes = (ulong)(rows * cols * sizeof(float));
            fixed (float* hostPtr = hostData)
            {
                var status = CublasNativeMethods.cudaMemcpy(
                    devicePtr, (IntPtr)hostPtr, sizeInBytes, CudaMemcpyKind.HostToDevice);
                CudaContext.CheckCudaStatus(status, "cudaMemcpy(H2D)");
            }
        }

        /// <summary>
        /// Copies a 1D float array from host to device.
        /// </summary>
        internal static unsafe void CopyToDevice(float[] hostData, IntPtr devicePtr, int count)
        {
            ulong sizeInBytes = (ulong)(count * sizeof(float));
            fixed (float* hostPtr = hostData)
            {
                var status = CublasNativeMethods.cudaMemcpy(
                    devicePtr, (IntPtr)hostPtr, sizeInBytes, CudaMemcpyKind.HostToDevice);
                CudaContext.CheckCudaStatus(status, "cudaMemcpy(H2D)");
            }
        }

        /// <summary>
        /// Copies from device to a 2D float array on host.
        /// </summary>
        internal static unsafe void CopyFromDevice(IntPtr devicePtr, float[,] hostData, int rows, int cols)
        {
            ulong sizeInBytes = (ulong)(rows * cols * sizeof(float));
            fixed (float* hostPtr = hostData)
            {
                var status = CublasNativeMethods.cudaMemcpy(
                    (IntPtr)hostPtr, devicePtr, sizeInBytes, CudaMemcpyKind.DeviceToHost);
                CudaContext.CheckCudaStatus(status, "cudaMemcpy(D2H)");
            }
        }

        /// <summary>
        /// Copies from device to a 1D float array on host.
        /// </summary>
        internal static unsafe void CopyFromDevice(IntPtr devicePtr, float[] hostData, int count)
        {
            ulong sizeInBytes = (ulong)(count * sizeof(float));
            fixed (float* hostPtr = hostData)
            {
                var status = CublasNativeMethods.cudaMemcpy(
                    (IntPtr)hostPtr, devicePtr, sizeInBytes, CudaMemcpyKind.DeviceToHost);
                CudaContext.CheckCudaStatus(status, "cudaMemcpy(D2H)");
            }
        }

        /// <summary>
        /// Allocates device memory and uploads a 2D array in one step.
        /// </summary>
        internal static IntPtr AllocateAndUpload(float[,] hostData, int rows, int cols)
        {
            IntPtr devicePtr = Allocate(rows * cols);
            CopyToDevice(hostData, devicePtr, rows, cols);
            return devicePtr;
        }

        /// <summary>
        /// Allocates device memory and uploads a 1D array in one step.
        /// </summary>
        internal static IntPtr AllocateAndUpload(float[] hostData)
        {
            IntPtr devicePtr = Allocate(hostData.Length);
            CopyToDevice(hostData, devicePtr, hostData.Length);
            return devicePtr;
        }

        /// <summary>
        /// Copies device-to-device memory.
        /// </summary>
        internal static void CopyDeviceToDevice(IntPtr src, IntPtr dst, int floatCount)
        {
            ulong sizeInBytes = (ulong)(floatCount * sizeof(float));
            var status = CublasNativeMethods.cudaMemcpy(dst, src, sizeInBytes, CudaMemcpyKind.DeviceToDevice);
            CudaContext.CheckCudaStatus(status, "cudaMemcpy(D2D)");
        }
    }
}
