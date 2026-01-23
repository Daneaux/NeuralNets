namespace MatrixLibrary
{
    /// <summary>
    /// GPU memory manager (stub - throws NotImplementedException).
    /// This stub allows the factory to work without requiring
    /// a full GPU implementation. In Phase 3, these methods will be
    /// replaced with actual CUDA/OpenCL implementations.
    /// </summary>
    public class GpuMemoryManager
    {
        /// <summary>
        /// Initializes the GPU device and runtime.
        /// </summary>
        public void Initialize()
        {
            throw new NotImplementedException("GPU backend not yet implemented - use AVX or Software backend");
        }

        /// <summary>
        /// Shuts down the GPU device and releases all resources.
        /// </summary>
        public void Shutdown()
        {
            throw new NotImplementedException("GPU backend not yet implemented - use AVX or Software backend");
        }

        /// <summary>
        /// Allocates memory on the GPU device.
        /// </summary>
        /// <param name="sizeInBytes">Size in bytes to allocate</param>
        /// <returns>Pointer to allocated device memory</returns>
        public IntPtr AllocateDeviceMemory(int sizeInBytes)
        {
            throw new NotImplementedException("GPU backend not yet implemented - use AVX or Software backend");
        }

        /// <summary>
        /// Frees memory on the GPU device.
        /// </summary>
        /// <param name="devicePtr">Pointer to device memory to free</param>
        public void FreeDeviceMemory(IntPtr devicePtr)
        {
            throw new NotImplementedException("GPU backend not yet implemented - use AVX or Software backend");
        }

        /// <summary>
        /// Copies data from host memory to device memory.
        /// </summary>
        /// <param name="hostData">Array containing data to copy</param>
        /// <param name="devicePtr">Pointer to device memory destination</param>
        public void CopyHostToDevice(float[] hostData, IntPtr devicePtr)
        {
            throw new NotImplementedException("GPU backend not yet implemented - use AVX or Software backend");
        }

        /// <summary>
        /// Copies data from device memory to host memory.
        /// </summary>
        /// <param name="devicePtr">Pointer to device memory source</param>
        /// <param name="hostData">Array to copy data into</param>
        public void CopyDeviceToHost(IntPtr devicePtr, float[] hostData)
        {
            throw new NotImplementedException("GPU backend not yet implemented - use AVX or Software backend");
        }

        /// <summary>
        /// Rents memory from a pool (for performance optimization).
        /// </summary>
        /// <param name="sizeInBytes">Size in bytes to rent</param>
        /// <returns>Pointer to pooled device memory</returns>
        public IntPtr RentMemory(int sizeInBytes)
        {
            throw new NotImplementedException("GPU backend not yet implemented - use AVX or Software backend");
        }

        /// <summary>
        /// Returns memory to the pool.
        /// </summary>
        /// <param name="devicePtr">Pointer to device memory to return</param>
        public void ReturnMemory(IntPtr devicePtr)
        {
            throw new NotImplementedException("GPU backend not yet implemented - use AVX or Software backend");
        }

        /// <summary>
        /// Sets the active GPU device.
        /// </summary>
        /// <param name="deviceId">Device ID to activate</param>
        public void SetActiveDevice(int deviceId)
        {
            throw new NotImplementedException("GPU backend not yet implemented - use AVX or Software backend");
        }

        /// <summary>
        /// Gets the active GPU device ID.
        /// </summary>
        /// <returns>Active device ID</returns>
        public int GetActiveDevice()
        {
            throw new NotImplementedException("GPU backend not yet implemented - use AVX or Software backend");
        }

        /// <summary>
        /// Gets the number of available GPU devices.
        /// </summary>
        /// <returns>Number of devices</returns>
        public int GetDeviceCount()
        {
            throw new NotImplementedException("GPU backend not yet implemented - use AVX or Software backend");
        }
    }
}
