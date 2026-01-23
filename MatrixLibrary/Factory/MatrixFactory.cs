using MatrixLibrary.BaseClasses;

namespace MatrixLibrary
{
    /// <summary>
    /// Factory for creating matrix and vector instances with configurable backend.
    /// </summary>
    public static class MatrixFactory
    {
        private static MatrixBackend _defaultBackend = MatrixBackend.AVX;

        /// <summary>
        /// Gets or sets the default backend for matrix creation.
        /// </summary>
        public static MatrixBackend DefaultBackend
        {
            get => _defaultBackend;
            set => _defaultBackend = value;
        }

        /// <summary>
        /// Sets the default backend for matrix creation.
        /// </summary>
        /// <param name="backend">The backend to use</param>
        public static void SetDefaultBackend(MatrixBackend backend)
        {
            _defaultBackend = backend;
        }

        /// <summary>
        /// Gets the current default backend.
        /// </summary>
        /// <returns>Current default backend</returns>
        public static MatrixBackend GetDefaultBackend()
        {
            return _defaultBackend;
        }

        // Matrix creation methods

        /// <summary>
        /// Creates a matrix with the specified dimensions using the default backend.
        /// </summary>
        /// <param name="rows">Number of rows</param>
        /// <param name="cols">Number of columns</param>
        /// <returns>New matrix instance</returns>
        public static MatrixBase CreateMatrix(int rows, int cols)
        {
            return CreateMatrix(rows, cols, _defaultBackend);
        }

        /// <summary>
        /// Creates a matrix from a 2D array using the default backend.
        /// </summary>
        /// <param name="data">2D array of float values</param>
        /// <returns>New matrix instance</returns>
        public static MatrixBase CreateMatrix(float[,] data)
        {
            return CreateMatrix(data, _defaultBackend);
        }

        /// <summary>
        /// Creates a matrix with the specified dimensions using the specified backend.
        /// </summary>
        /// <param name="rows">Number of rows</param>
        /// <param name="cols">Number of columns</param>
        /// <param name="backend">Backend implementation to use</param>
        /// <returns>New matrix instance</returns>
        public static MatrixBase CreateMatrix(int rows, int cols, MatrixBackend backend)
        {
            return backend switch
            {
                MatrixBackend.Software => new Matrix2D(rows, cols),
                MatrixBackend.AVX => new AvxMatrix(rows, cols),
                MatrixBackend.GPU => CreateGpuMatrix(rows, cols),
                _ => throw new ArgumentException($"Unsupported backend: {backend}")
            };
        }

        /// <summary>
        /// Creates a matrix from a 2D array using the specified backend.
        /// </summary>
        /// <param name="data">2D array of float values</param>
        /// <param name="backend">Backend implementation to use</param>
        /// <returns>New matrix instance</returns>
        public static MatrixBase CreateMatrix(float[,] data, MatrixBackend backend)
        {
            return backend switch
            {
                MatrixBackend.Software => new Matrix2D(data),
                MatrixBackend.AVX => new AvxMatrix(data),
                MatrixBackend.GPU => CreateGpuMatrix(data),
                _ => throw new ArgumentException($"Unsupported backend: {backend}")
            };
        }

        // Column vector creation methods

        /// <summary>
        /// Creates a column vector with the specified size using the default backend.
        /// </summary>
        /// <param name="size">Number of elements</param>
        /// <returns>New column vector instance</returns>
        public static BaseClasses.ColumnVectorBase CreateColumnVector(int size)
        {
            return CreateColumnVector(size, _defaultBackend);
        }

        /// <summary>
        /// Creates a column vector from an array using the default backend.
        /// </summary>
        /// <param name="data">Array of float values</param>
        /// <returns>New column vector instance</returns>
        public static BaseClasses.ColumnVectorBase CreateColumnVector(float[] data)
        {
            return CreateColumnVector(data, _defaultBackend);
        }

        /// <summary>
        /// Creates a column vector with the specified size using the specified backend.
        /// </summary>
        /// <param name="size">Number of elements</param>
        /// <param name="backend">Backend implementation to use</param>
        /// <returns>New column vector instance</returns>
        public static ColumnVectorBase CreateColumnVector(int size, MatrixBackend backend)
        {
            return backend switch
            {
                MatrixBackend.Software => new ColumnVector(size),
                MatrixBackend.AVX => new AvxColumnVector(size),
                MatrixBackend.GPU => CreateGpuColumnVector(size),
                _ => throw new ArgumentException($"Unsupported backend: {backend}")
            };
        }

        /// <summary>
        /// Creates a column vector from an array using the specified backend.
        /// </summary>
        /// <param name="data">Array of float values</param>
        /// <param name="backend">Backend implementation to use</param>
        /// <returns>New column vector instance</returns>
        public static ColumnVectorBase CreateColumnVector(float[] data, MatrixBackend backend)
        {
            return backend switch
            {
                MatrixBackend.Software => new ColumnVector(data),
                MatrixBackend.AVX => new AvxColumnVector(data),
                MatrixBackend.GPU => CreateGpuColumnVector(data),
                _ => throw new ArgumentException($"Unsupported backend: {backend}")
            };
        }

        // Row vector creation methods

        /// <summary>
        /// Creates a row vector with the specified size using the default backend.
        /// </summary>
        /// <param name="size">Number of elements</param>
        /// <returns>New row vector instance</returns>
        public static RowVectorBase CreateRowVector(int size)
        {
            return CreateRowVector(size, _defaultBackend);
        }

        /// <summary>
        /// Creates a row vector from an array using the default backend.
        /// </summary>
        /// <param name="data">Array of float values</param>
        /// <returns>New row vector instance</returns>
        public static RowVectorBase CreateRowVector(float[] data)
        {
            return CreateRowVector(data, _defaultBackend);
        }

        /// <summary>
        /// Creates a row vector with the specified size using the specified backend.
        /// </summary>
        /// <param name="size">Number of elements</param>
        /// <param name="backend">Backend implementation to use</param>
        /// <returns>New row vector instance</returns>
        public static RowVectorBase CreateRowVector(int size, MatrixBackend backend)
        {
            return backend switch
            {
                MatrixBackend.Software => new RowVector(size),
                MatrixBackend.AVX => throw new NotSupportedException("RowVector not implemented for AVX backend"),
                MatrixBackend.GPU => CreateGpuRowVector(size),
                _ => throw new ArgumentException($"Unsupported backend: {backend}")
            };
        }

        /// <summary>
        /// Creates a row vector from an array using the specified backend.
        /// </summary>
        /// <param name="data">Array of float values</param>
        /// <param name="backend">Backend implementation to use</param>
        /// <returns>New row vector instance</returns>
        public static RowVectorBase CreateRowVector(float[] data, MatrixBackend backend)
        {
            return backend switch
            {
                MatrixBackend.Software => new RowVector(data),
                MatrixBackend.AVX => throw new NotSupportedException("RowVector not implemented for AVX backend"),
                MatrixBackend.GPU => CreateGpuRowVector(data),
                _ => throw new ArgumentException($"Unsupported backend: {backend}")
            };
        }

        // GPU stub creation methods (to be implemented in Phase 3)

        private static MatrixBase CreateGpuMatrix(int rows, int cols)
        {
            return new GpuMatrix(rows, cols);
        }

        private static MatrixBase CreateGpuMatrix(float[,] data)
        {
            return new GpuMatrix(data);
        }

        private static ColumnVectorBase CreateGpuColumnVector(int size)
        {
            return new GpuColumnVector(size);
        }

        private static ColumnVectorBase CreateGpuColumnVector(float[] data)
        {
            return new GpuColumnVector(data);
        }

        private static RowVectorBase CreateGpuRowVector(int size)
        {
            return new GpuRowVector(size);
        }

        private static RowVectorBase CreateGpuRowVector(float[] data)
        {
            return new GpuRowVector(data);
        }
    }
}
