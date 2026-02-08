using MatrixLibrary.BaseClasses;
using MatrixLibrary.Gpu;

namespace MatrixLibrary
{
    /// <summary>
    /// GPU-accelerated matrix implementation using CUBLAS.
    ///
    /// Row-major/Column-major strategy:
    /// C# float[,] is row-major. CUBLAS expects column-major. A row-major matrix
    /// interpreted as column-major IS its transpose. We exploit this by swapping
    /// operand order in cublasSgemm calls, avoiding any explicit transpose of data.
    ///
    /// Memory model: lazy device-resident.
    ///
    /// Each GpuMatrix maintains both a host buffer (Mat) and a device pointer (_devicePtr).
    /// Data can be valid on one side, both, or neither (freshly allocated).
    ///
    /// - _deviceValid: device has current data (no need to upload)
    /// - _hostValid:   host Mat has current data (no need to download)
    ///
    /// GPU operations produce results that live ONLY on the device (_hostValid=false).
    /// Host data is only downloaded when someone reads Mat or the indexer.
    /// This means chained GPU ops (e.g., A.Add(B).Multiply(C)) never round-trip through host.
    /// </summary>
    public class GpuMatrix : MatrixBase, IDisposable
    {
        private IntPtr _devicePtr = IntPtr.Zero;
        private bool _deviceValid = false;
        private bool _hostValid = true;

        /// <summary>
        /// Overridden Mat property: downloads device data to host on first access.
        /// </summary>
        public override float[,] Mat
        {
            get
            {
                EnsureHostUpToDate();
                return base.Mat;
            }
            protected set { base.Mat = value; }
        }

        /// <summary>
        /// Overridden indexer: downloads device data to host on first read access.
        /// </summary>
        public override float this[int r, int c]
        {
            get
            {
                EnsureHostUpToDate();
                return base.Mat[r, c];
            }
            set
            {
                EnsureHostUpToDate();
                base.Mat[r, c] = value;
                InvalidateDevice();
            }
        }

        public GpuMatrix(int rows, int cols)
        {
            Rows = rows;
            Cols = cols;
            base.Mat = new float[rows, cols];
            _hostValid = true;
        }

        public GpuMatrix(float[,] data)
        {
            base.Mat = (float[,])data.Clone();
            Rows = data.GetLength(0);
            Cols = data.GetLength(1);
            _hostValid = true;
        }

        /// <summary>
        /// Internal constructor: device-only result. Host array is allocated but NOT populated.
        /// Data lives only on GPU until someone reads Mat or the indexer.
        /// </summary>
        internal GpuMatrix(int rows, int cols, IntPtr devicePtr)
        {
            Rows = rows;
            Cols = cols;
            base.Mat = new float[rows, cols]; // allocated but uninitialized
            _devicePtr = devicePtr;
            _deviceValid = true;
            _hostValid = false; // data is ONLY on device
        }

        /// <summary>
        /// Internal constructor that takes ownership of host data and a device pointer.
        /// Both host and device are valid.
        /// </summary>
        internal GpuMatrix(float[,] data, IntPtr devicePtr)
        {
            base.Mat = data;
            Rows = data.GetLength(0);
            Cols = data.GetLength(1);
            _devicePtr = devicePtr;
            _deviceValid = true;
            _hostValid = true;
        }

        /// <summary>
        /// Downloads device data to host if not already valid.
        /// </summary>
        private void EnsureHostUpToDate()
        {
            if (!_hostValid && _deviceValid && _devicePtr != IntPtr.Zero)
            {
                GpuMemoryManager.CopyFromDevice(_devicePtr, base.Mat, Rows, Cols);
                _hostValid = true;
            }
        }

        /// <summary>
        /// Ensures device memory is allocated and contains current data.
        /// If host was modified, uploads to device.
        /// </summary>
        internal IntPtr EnsureDeviceUpToDate()
        {
            if (_devicePtr == IntPtr.Zero)
            {
                _devicePtr = GpuMemoryManager.Allocate(Rows * Cols);
                _deviceValid = false;
            }
            if (!_deviceValid)
            {
                // Host must be valid if device isn't
                GpuMemoryManager.CopyToDevice(base.Mat, _devicePtr, Rows, Cols);
                _deviceValid = true;
            }
            return _devicePtr;
        }

        /// <summary>
        /// Gets device pointer for another MatrixBase (uploads if needed).
        /// If it's a GpuMatrix, reuses its cached device pointer.
        /// Otherwise allocates and uploads.
        /// Caller must free returned pointer if allocatedPtr != IntPtr.Zero.
        /// </summary>
        private static IntPtr GetDevicePtr(MatrixBase other, out IntPtr allocatedPtr)
        {
            if (other is GpuMatrix gpu)
            {
                allocatedPtr = IntPtr.Zero;
                return gpu.EnsureDeviceUpToDate();
            }
            else
            {
                allocatedPtr = GpuMemoryManager.AllocateAndUpload(other.Mat, other.Rows, other.Cols);
                return allocatedPtr;
            }
        }

        /// <summary>
        /// Invalidates the device cache (call after host data is modified).
        /// </summary>
        private void InvalidateDevice()
        {
            _deviceValid = false;
        }

        // ============================================================
        // CUBLAS-accelerated operations
        // ============================================================

        /// <summary>
        /// Matrix multiplication using cublasSgemm.
        /// Row-major trick: swap A and B in the call.
        /// cublasSgemm(N, N, B.Cols, A.Rows, A.Cols, 1, d_B, B.Cols, d_A, A.Cols, 0, d_C, B.Cols)
        /// </summary>
        public override unsafe MatrixBase Multiply(MatrixBase other)
        {
            if (this.Cols != other.Rows)
                throw new ArgumentException($"Matrix dimension mismatch: {Rows}x{Cols} * {other.Rows}x{other.Cols}");

            var handle = CudaContext.Instance.CublasHandle;
            IntPtr dA = EnsureDeviceUpToDate();
            IntPtr dB_other = GetDevicePtr(other, out IntPtr allocatedB);

            int resultRows = this.Rows;
            int resultCols = other.Cols;
            IntPtr dC = GpuMemoryManager.Allocate(resultRows * resultCols);

            try
            {
                float alpha = 1.0f;
                float beta = 0.0f;

                // Row-major trick: call gemm with B, A swapped
                // m = other.Cols, n = this.Rows, k = this.Cols
                var status = CublasNativeMethods.cublasSgemm_v2(
                    handle,
                    CublasOperation.None,   // transa (for B seen as B^T)
                    CublasOperation.None,   // transb (for A seen as A^T)
                    other.Cols,             // m: rows of op(B^T) = cols of B
                    this.Rows,              // n: cols of op(A^T) = rows of A
                    this.Cols,              // k: shared dimension
                    &alpha,
                    dB_other, other.Cols,   // A in cublas = B (our), lda = B.Cols
                    dA, this.Cols,          // B in cublas = A (our), ldb = A.Cols
                    &beta,
                    dC, other.Cols);        // C, ldc = result cols = B.Cols

                CudaContext.CheckCublasStatus(status, "cublasSgemm_v2");

                return new GpuMatrix(resultRows, resultCols, dC);
            }
            catch
            {
                GpuMemoryManager.Free(dC);
                throw;
            }
            finally
            {
                if (allocatedB != IntPtr.Zero)
                    GpuMemoryManager.Free(allocatedB);
            }
        }

        /// <summary>
        /// Matrix-vector multiplication using cublasSgemv.
        /// Row-major trick: our row-major A looks like A^T to CUBLAS, so use OP_T.
        /// cublasSgemv(OP_T, A.Cols, A.Rows, 1, d_A, A.Cols, d_x, 1, 0, d_y, 1)
        /// </summary>
        public override unsafe ColumnVectorBase MatrixTimesColumn(ColumnVectorBase column)
        {
            if (this.Cols != column.Size)
                throw new ArgumentException($"Dimension mismatch: {Rows}x{Cols} matrix * {column.Size} vector");

            var handle = CudaContext.Instance.CublasHandle;
            IntPtr dA = EnsureDeviceUpToDate();

            IntPtr dX;
            IntPtr allocatedX = IntPtr.Zero;
            if (column is GpuColumnVector gpuVec)
            {
                dX = gpuVec.EnsureDeviceUpToDate();
            }
            else
            {
                dX = GpuMemoryManager.AllocateAndUpload(column.Column);
                allocatedX = dX;
            }

            IntPtr dY = GpuMemoryManager.Allocate(this.Rows);

            try
            {
                float alpha = 1.0f;
                float beta = 0.0f;

                // Row-major trick: matrix stored as A^T from CUBLAS perspective
                // Use OP_T to undo: computes A^T^T * x = A * x
                var status = CublasNativeMethods.cublasSgemv_v2(
                    handle,
                    CublasOperation.Transpose,  // OP_T undoes implicit transpose
                    this.Cols,                  // m: rows of the column-major matrix (= our Cols)
                    this.Rows,                  // n: cols of the column-major matrix (= our Rows)
                    &alpha,
                    dA, this.Cols,              // A, lda = Cols (leading dim in col-major view)
                    dX, 1,                      // x, incx
                    &beta,
                    dY, 1);                     // y, incy

                CudaContext.CheckCublasStatus(status, "cublasSgemv_v2");

                float[] result = new float[this.Rows];
                GpuMemoryManager.CopyFromDevice(dY, result, this.Rows);

                var resultVec = new GpuColumnVector(result, dY);
                dY = IntPtr.Zero; // ownership transferred
                return resultVec;
            }
            catch
            {
                GpuMemoryManager.Free(dY);
                throw;
            }
            finally
            {
                if (allocatedX != IntPtr.Zero)
                    GpuMemoryManager.Free(allocatedX);
            }
        }

        /// <summary>
        /// Matrix addition using cublasSgeam: C = alpha*A + beta*B
        /// Element-wise, so row/col order doesn't matter (treated as flat).
        /// We use cublasSgeam with both matrices as non-transposed in their col-major view.
        /// </summary>
        public override unsafe MatrixBase Add(MatrixBase other)
        {
            if (this.Rows != other.Rows || this.Cols != other.Cols)
                throw new ArgumentException($"Dimension mismatch: {Rows}x{Cols} + {other.Rows}x{other.Cols}");

            var handle = CudaContext.Instance.CublasHandle;
            IntPtr dA = EnsureDeviceUpToDate();
            IntPtr dB = GetDevicePtr(other, out IntPtr allocatedB);
            IntPtr dC = GpuMemoryManager.Allocate(Rows * Cols);

            try
            {
                float alpha = 1.0f;
                float beta = 1.0f;

                // cublasSgeam: C = alpha*op(A) + beta*op(B)
                // For element-wise add, both ops are None.
                // m and n are dimensions of the col-major view of our row-major data.
                // Our (Rows x Cols) row-major = (Cols x Rows) col-major.
                var status = CublasNativeMethods.cublasSgeam(
                    handle,
                    CublasOperation.None, CublasOperation.None,
                    this.Cols,  // m: rows in col-major view
                    this.Rows,  // n: cols in col-major view
                    &alpha,
                    dA, this.Cols,   // lda
                    &beta,
                    dB, this.Cols,   // ldb
                    dC, this.Cols);  // ldc

                CudaContext.CheckCublasStatus(status, "cublasSgeam(Add)");

                return new GpuMatrix(Rows, Cols, dC);
            }
            catch
            {
                GpuMemoryManager.Free(dC);
                throw;
            }
            finally
            {
                if (allocatedB != IntPtr.Zero)
                    GpuMemoryManager.Free(allocatedB);
            }
        }

        /// <summary>
        /// Matrix subtraction using cublasSgeam: C = 1*A + (-1)*B
        /// </summary>
        public override unsafe MatrixBase Subtract(MatrixBase other)
        {
            if (this.Rows != other.Rows || this.Cols != other.Cols)
                throw new ArgumentException($"Dimension mismatch: {Rows}x{Cols} - {other.Rows}x{other.Cols}");

            var handle = CudaContext.Instance.CublasHandle;
            IntPtr dA = EnsureDeviceUpToDate();
            IntPtr dB = GetDevicePtr(other, out IntPtr allocatedB);
            IntPtr dC = GpuMemoryManager.Allocate(Rows * Cols);

            try
            {
                float alpha = 1.0f;
                float beta = -1.0f;

                var status = CublasNativeMethods.cublasSgeam(
                    handle,
                    CublasOperation.None, CublasOperation.None,
                    this.Cols, this.Rows,
                    &alpha,
                    dA, this.Cols,
                    &beta,
                    dB, this.Cols,
                    dC, this.Cols);

                CudaContext.CheckCublasStatus(status, "cublasSgeam(Subtract)");

                return new GpuMatrix(Rows, Cols, dC);
            }
            catch
            {
                GpuMemoryManager.Free(dC);
                throw;
            }
            finally
            {
                if (allocatedB != IntPtr.Zero)
                    GpuMemoryManager.Free(allocatedB);
            }
        }

        /// <summary>
        /// Scalar multiplication using cublasSscal on a copy of the data.
        /// </summary>
        public override unsafe MatrixBase Multiply(float scalar)
        {
            var handle = CudaContext.Instance.CublasHandle;

            // Copy device data to a new buffer (don't modify in-place)
            IntPtr dA = EnsureDeviceUpToDate();
            IntPtr dResult = GpuMemoryManager.Allocate(Rows * Cols);
            GpuMemoryManager.CopyDeviceToDevice(dA, dResult, Rows * Cols);

            try
            {
                var status = CublasNativeMethods.cublasSscal_v2(
                    handle,
                    Rows * Cols,
                    &scalar,
                    dResult, 1);

                CudaContext.CheckCublasStatus(status, "cublasSscal_v2");

                return new GpuMatrix(Rows, Cols, dResult);
            }
            catch
            {
                GpuMemoryManager.Free(dResult);
                throw;
            }
        }

        /// <summary>
        /// Transpose using cublasSgeam: C = alpha * A^T + beta * B
        /// With beta=0, this is just C = A^T.
        /// The row-major/col-major trick makes this elegant:
        /// Our row-major (Rows x Cols) = col-major (Cols x Rows).
        /// Transposing in col-major gives (Rows x Cols) col-major = (Cols x Rows) row-major.
        /// </summary>
        public override unsafe MatrixBase Transpose()
        {
            return GetTransposedMatrix();
        }

        public override unsafe MatrixBase GetTransposedMatrix()
        {
            var handle = CudaContext.Instance.CublasHandle;
            IntPtr dA = EnsureDeviceUpToDate();

            int resultRows = Cols;
            int resultCols = Rows;
            IntPtr dC = GpuMemoryManager.Allocate(resultRows * resultCols);

            try
            {
                float alpha = 1.0f;
                float beta = 0.0f;

                // Our row-major A (Rows x Cols) is seen as col-major (Cols x Rows) by CUBLAS.
                // We want the transpose of A: (Cols x Rows) in row-major = (Rows x Cols) in col-major.
                // cublasSgeam with OP_T on A:
                //   C (m x n col-major) = alpha * A^T (m x n) + beta * B
                //   where A is (n x m) col-major, so A^T is (m x n) col-major
                //   m = Rows (result rows in col-major = result cols in row-major = original Rows)
                //   n = Cols (result cols in col-major = result rows in row-major = original Cols)
                // Wait — let's think carefully:
                //   Our row-major A[Rows][Cols] = col-major matrix with m=Cols, n=Rows.
                //   We want result row-major [Cols][Rows] = col-major with m=Rows, n=Cols.
                //   So: C(m=Rows, n=Cols) = A^T where A is (Cols x Rows) col-major.
                //   cublasSgeam(OP_T, OP_N, m=Rows, n=Cols, alpha, A, lda=Cols, beta, B=null, ldb=Rows, C, ldc=Rows)
                var status = CublasNativeMethods.cublasSgeam(
                    handle,
                    CublasOperation.Transpose, CublasOperation.None,
                    Rows,                   // m: rows of result in col-major
                    Cols,                   // n: cols of result in col-major
                    &alpha,
                    dA, Cols,               // A, lda = Cols (leading dim of A in col-major)
                    &beta,
                    dA, Rows,               // B ignored (beta=0), ldb must still be valid
                    dC, Rows);              // C, ldc = Rows (leading dim of C in col-major)

                CudaContext.CheckCublasStatus(status, "cublasSgeam(Transpose)");

                return new GpuMatrix(resultRows, resultCols, dC);
            }
            catch
            {
                GpuMemoryManager.Free(dC);
                throw;
            }
        }

        // ============================================================
        // Fallback operations (use AvxMatrix on host)
        // ============================================================

        public override MatrixBase Add(float scalar)
        {
            var avx = new AvxMatrix(this.Mat);
            var result = avx.Add(scalar);
            return new GpuMatrix(result.Mat);
        }

        public override MatrixBase HadamardProduct(MatrixBase other)
        {
            var avx = new AvxMatrix(this.Mat);
            var otherMat = (other is GpuMatrix g) ? g.Mat : other.Mat;
            var result = avx.HadamardProduct(new AvxMatrix(otherMat));
            return new GpuMatrix(result.Mat);
        }

        public override MatrixBase Log()
        {
            var avx = new AvxMatrix(this.Mat);
            var result = avx.Log();
            return new GpuMatrix(result.Mat);
        }

        public override void SetDiagonal(float diagonalValue)
        {
            InvalidateDevice();
            int n = Math.Min(Rows, Cols);
            for (int i = 0; i < n; i++)
            {
                Mat[i, i] = diagonalValue;
            }
        }

        public override MatrixBase Convolution(MatrixBase kernel)
        {
            var avx = new AvxMatrix(this.Mat);
            var kernelMat = (kernel is GpuMatrix g) ? g.Mat : kernel.Mat;
            var result = avx.Convolution(new AvxMatrix(kernelMat));
            return new GpuMatrix(result.Mat);
        }

        public override MatrixBase ConvolutionFull(MatrixBase kernel)
        {
            var avx = new AvxMatrix(this.Mat);
            var kernelMat = (kernel is GpuMatrix g) ? g.Mat : kernel.Mat;
            var result = avx.ConvolutionFull(new AvxMatrix(kernelMat));
            return new GpuMatrix(result.Mat);
        }

        public override float Sum()
        {
            var avx = new AvxMatrix(this.Mat);
            return avx.Sum();
        }

        // ============================================================
        // IDisposable - free device memory
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

        ~GpuMatrix()
        {
            if (!_disposed && _devicePtr != IntPtr.Zero)
            {
                // Best-effort free in finalizer — CUDA context may be gone
                try { GpuMemoryManager.Free(_devicePtr); } catch { }
                _devicePtr = IntPtr.Zero;
            }
        }
    }
}
