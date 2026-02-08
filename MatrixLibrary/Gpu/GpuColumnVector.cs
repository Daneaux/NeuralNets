using MatrixLibrary.BaseClasses;
using MatrixLibrary.Gpu;

namespace MatrixLibrary
{
    /// <summary>
    /// GPU-accelerated column vector implementation using CUBLAS.
    /// Caches device memory for reuse across operations.
    /// </summary>
    public class GpuColumnVector : ColumnVectorBase, IDisposable
    {
        private IntPtr _devicePtr = IntPtr.Zero;
        private bool _deviceValid = false;

        public GpuColumnVector(int size) : base(size) { }

        public GpuColumnVector(float[] data) : base((float[])data.Clone()) { }

        /// <summary>
        /// Internal constructor that takes ownership of data and a pre-allocated device pointer.
        /// </summary>
        internal GpuColumnVector(float[] data, IntPtr devicePtr) : base(data)
        {
            _devicePtr = devicePtr;
            _deviceValid = true;
        }

        /// <summary>
        /// Ensures device memory is allocated and contains current host data.
        /// </summary>
        internal IntPtr EnsureDeviceUpToDate()
        {
            if (_devicePtr == IntPtr.Zero)
            {
                _devicePtr = GpuMemoryManager.Allocate(Size);
                _deviceValid = false;
            }
            if (!_deviceValid)
            {
                GpuMemoryManager.CopyToDevice(column, _devicePtr, Size);
                _deviceValid = true;
            }
            return _devicePtr;
        }

        private static IntPtr GetDevicePtr(ColumnVectorBase other, out IntPtr allocatedPtr)
        {
            if (other is GpuColumnVector gpu)
            {
                allocatedPtr = IntPtr.Zero;
                return gpu.EnsureDeviceUpToDate();
            }
            else
            {
                allocatedPtr = GpuMemoryManager.AllocateAndUpload(other.Column);
                return allocatedPtr;
            }
        }

        private void InvalidateDevice()
        {
            _deviceValid = false;
        }

        // ============================================================
        // CUBLAS-accelerated operations
        // ============================================================

        /// <summary>
        /// Vector addition using cublasSaxpy: y = alpha*x + y (with alpha=1).
        /// We copy this to result, then axpy rhs onto result.
        /// </summary>
        public override unsafe ColumnVectorBase Add(ColumnVectorBase rhs)
        {
            if (this.Size != rhs.Size)
                throw new ArgumentException($"Vector size mismatch: {Size} vs {rhs.Size}");

            var handle = CudaContext.Instance.CublasHandle;

            // Result = copy of this
            IntPtr dThis = EnsureDeviceUpToDate();
            IntPtr dResult = GpuMemoryManager.Allocate(Size);
            GpuMemoryManager.CopyDeviceToDevice(dThis, dResult, Size);

            IntPtr dRhs = GetDevicePtr(rhs, out IntPtr allocatedRhs);

            try
            {
                float alpha = 1.0f;

                var status = CublasNativeMethods.cublasSaxpy_v2(
                    handle, Size, &alpha, dRhs, 1, dResult, 1);
                CudaContext.CheckCublasStatus(status, "cublasSaxpy_v2(Add)");

                float[] result = new float[Size];
                GpuMemoryManager.CopyFromDevice(dResult, result, Size);
                return new GpuColumnVector(result, dResult);
            }
            catch
            {
                GpuMemoryManager.Free(dResult);
                throw;
            }
            finally
            {
                if (allocatedRhs != IntPtr.Zero)
                    GpuMemoryManager.Free(allocatedRhs);
            }
        }

        /// <summary>
        /// Vector subtraction: this - rhs = this + (-1)*rhs
        /// Copy this to result, then axpy with alpha=-1.
        /// </summary>
        public override unsafe ColumnVectorBase Subtract(ColumnVectorBase rhs)
        {
            if (this.Size != rhs.Size)
                throw new ArgumentException($"Vector size mismatch: {Size} vs {rhs.Size}");

            var handle = CudaContext.Instance.CublasHandle;

            IntPtr dThis = EnsureDeviceUpToDate();
            IntPtr dResult = GpuMemoryManager.Allocate(Size);
            GpuMemoryManager.CopyDeviceToDevice(dThis, dResult, Size);

            IntPtr dRhs = GetDevicePtr(rhs, out IntPtr allocatedRhs);

            try
            {
                float alpha = -1.0f;

                var status = CublasNativeMethods.cublasSaxpy_v2(
                    handle, Size, &alpha, dRhs, 1, dResult, 1);
                CudaContext.CheckCublasStatus(status, "cublasSaxpy_v2(Subtract)");

                float[] result = new float[Size];
                GpuMemoryManager.CopyFromDevice(dResult, result, Size);
                return new GpuColumnVector(result, dResult);
            }
            catch
            {
                GpuMemoryManager.Free(dResult);
                throw;
            }
            finally
            {
                if (allocatedRhs != IntPtr.Zero)
                    GpuMemoryManager.Free(allocatedRhs);
            }
        }

        /// <summary>
        /// Scalar multiplication using cublasSscal.
        /// </summary>
        public override unsafe ColumnVectorBase Multiply(float scalar)
        {
            var handle = CudaContext.Instance.CublasHandle;

            IntPtr dThis = EnsureDeviceUpToDate();
            IntPtr dResult = GpuMemoryManager.Allocate(Size);
            GpuMemoryManager.CopyDeviceToDevice(dThis, dResult, Size);

            try
            {
                var status = CublasNativeMethods.cublasSscal_v2(
                    handle, Size, &scalar, dResult, 1);
                CudaContext.CheckCublasStatus(status, "cublasSscal_v2");

                float[] result = new float[Size];
                GpuMemoryManager.CopyFromDevice(dResult, result, Size);
                return new GpuColumnVector(result, dResult);
            }
            catch
            {
                GpuMemoryManager.Free(dResult);
                throw;
            }
        }

        /// <summary>
        /// Outer product using cublasSger: A = alpha * x * y^T
        /// Row-major trick: swap x and y so col-major result = row-major outer product.
        /// Result is (this.Size x rhs.Size) matrix.
        /// </summary>
        public override unsafe MatrixBase OuterProduct(ColumnVectorBase rhs)
        {
            var handle = CudaContext.Instance.CublasHandle;

            IntPtr dThis = EnsureDeviceUpToDate();
            IntPtr dRhs = GetDevicePtr(rhs, out IntPtr allocatedRhs);

            int resultRows = this.Size;
            int resultCols = rhs.Size;
            IntPtr dA = GpuMemoryManager.Allocate(resultRows * resultCols);

            try
            {
                // Zero the result matrix (cublasSger does A = alpha*x*y^T + A)
                float[,] zeros = new float[resultRows, resultCols];
                GpuMemoryManager.CopyToDevice(zeros, dA, resultRows, resultCols);

                float alpha = 1.0f;

                // Row-major trick for outer product:
                // We want result[i,j] = this[i] * rhs[j] in row-major.
                // cublasSger produces A[i,j] = x[i]*y[j] in COLUMN-major.
                // Column-major A with m=rhs.Size, n=this.Size gives:
                //   A_colmajor[i,j] = rhs[i] * this[j]
                // which, read as row-major with dims (this.Size x rhs.Size), IS this[row]*rhs[col].
                var status = CublasNativeMethods.cublasSger_v2(
                    handle,
                    rhs.Size,       // m: rows of col-major result
                    this.Size,      // n: cols of col-major result
                    &alpha,
                    dRhs, 1,        // x = rhs (first vector in col-major)
                    dThis, 1,       // y = this (second vector in col-major)
                    dA, rhs.Size);  // A, lda = m = rhs.Size

                CudaContext.CheckCublasStatus(status, "cublasSger_v2");

                float[,] result = new float[resultRows, resultCols];
                GpuMemoryManager.CopyFromDevice(dA, result, resultRows, resultCols);
                return new GpuMatrix(result, dA);
            }
            catch
            {
                GpuMemoryManager.Free(dA);
                throw;
            }
            finally
            {
                if (allocatedRhs != IntPtr.Zero)
                    GpuMemoryManager.Free(allocatedRhs);
            }
        }

        // ============================================================
        // Fallback operations (use AvxColumnVector on host)
        // ============================================================

        public override float Sum()
        {
            var avx = new AvxColumnVector(this.column);
            return avx.Sum();
        }

        public override ColumnVectorBase Log()
        {
            var avx = new AvxColumnVector(this.column);
            var result = avx.Log();
            return new GpuColumnVector(result.Column);
        }

        public override ColumnVectorBase Multiply(ColumnVectorBase rhs)
        {
            // Element-wise multiply — no single CUBLAS op, fall back to AVX
            var avx = new AvxColumnVector(this.column);
            var rhsData = (rhs is GpuColumnVector g) ? g.column : rhs.Column;
            var result = avx.Multiply(new AvxColumnVector(rhsData));
            return new GpuColumnVector(result.Column);
        }

        public override ColumnVectorBase Add(float scalar)
        {
            var avx = new AvxColumnVector(this.column);
            var result = avx.Add(scalar);
            return new GpuColumnVector(result.Column);
        }

        public override ColumnVectorBase Subtract(float scalar)
        {
            var avx = new AvxColumnVector(this.column);
            var result = avx.Subtract(scalar);
            return new GpuColumnVector(result.Column);
        }

        public override MatrixBase RhsOuterProduct(Tensor lhs)
        {
            var lhsVec = lhs.ToColumnVector();
            if (lhsVec == null)
                throw new ArgumentException("Tensor does not contain a column vector");
            return lhsVec.OuterProduct(this);
        }

        public override MatrixBase OuterProduct(FlattenedMatricesAsVector rhs)
        {
            // Flatten the matrices into a single array, then call cublasSger
            int totalSize = rhs.Size;
            float[] flattened = new float[totalSize];
            int offset = 0;
            foreach (var mat in rhs.Matrices)
            {
                int matSize = mat.TotalSize;
                Buffer.BlockCopy(mat.Mat, 0, flattened, offset * sizeof(float), matSize * sizeof(float));
                offset += matSize;
            }

            var flatVec = new GpuColumnVector(flattened);
            return OuterProduct(flatVec);
        }

        // ============================================================
        // IDisposable
        // ============================================================

        private bool _disposed = false;

        public void Dispose()
        {
            if (!_disposed)
            {
                GpuMemoryManager.Free(_devicePtr);
                _devicePtr = IntPtr.Zero;
                _deviceValid = false;
                _disposed = true;
            }
            GC.SuppressFinalize(this);
        }

        ~GpuColumnVector()
        {
            if (!_disposed && _devicePtr != IntPtr.Zero)
            {
                try { GpuMemoryManager.Free(_devicePtr); } catch { }
                _devicePtr = IntPtr.Zero;
            }
        }
    }
}
