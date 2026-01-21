using MatrixLib.Interfaces;

namespace MatrixLib
{
    public interface IMatrix
    {
        /// <summary>
        /// Number of rows in the matrix.
        /// </summary>
        int Rows { get; }

        /// <summary>
        /// Number of columns in the matrix.
        /// </summary>
        int Cols { get; }

        /// <summary>
        /// Total number of elements in the matrix.
        /// </summary>
        int TotalSize { get; }

        /// <summary>
        /// Gets or sets the value at the specified row and column.
        /// </summary>
        /// <param name="r">Row index (0-based)</param>
        /// <param name="c">Column index (0-based)</param>
        /// <returns>Value at position (r, c)</returns>
        float this[int r, int c] { get; set; }

        /// <summary>
        /// Gets the underlying data array (for compatibility with existing code).
        /// </summary>
        float[,] Mat { get; }

        /// <summary>
        /// Backend implementation type.
        /// </summary>
        //MatrixBackend Backend { get; }

        // Arithmetic operations

        /// <summary>
        /// Adds another matrix to this matrix element-wise.
        /// </summary>
        /// <param name="other">Matrix to add</param>
        /// <returns>New matrix containing the sum</returns>
        IMatrix Add(IMatrix other);

        /// <summary>
        /// Subtracts another matrix from this matrix element-wise.
        /// </summary>
        /// <param name="other">Matrix to subtract</param>
        /// <returns>New matrix containing the difference</returns>
        IMatrix Subtract(IMatrix other);

        /// <summary>
        /// Multiplies this matrix by another matrix (matrix multiplication).
        /// </summary>
        /// <param name="other">Matrix to multiply by</param>
        /// <returns>New matrix containing the product</returns>
        IMatrix Multiply(IMatrix other);

        /// <summary>
        /// Multiplies this matrix by a scalar.
        /// </summary>
        /// <param name="scalar">Scalar value to multiply by</param>
        /// <returns>New matrix with all elements multiplied by scalar</returns>
        IMatrix Multiply(float scalar);

        // Matrix-vector operations

        /// <summary>
        /// Multiplies this matrix by a column vector.
        /// </summary>
        /// <param name="vector">Column vector to multiply by</param>
        /// <returns>Resulting column vector</returns>
        IColumnVector Multiply(IColumnVector vector);

        // Utility operations

        /// <summary>
        /// Transposes this matrix (rows become columns, columns become rows).
        /// </summary>
        /// <returns>New transposed matrix</returns>
        IMatrix Transpose();

        /// <summary>
        /// Computes the natural logarithm of each element.
        /// </summary>
        /// <returns>New matrix with log of each element</returns>
        IMatrix Log();

        /// <summary>
        /// Computes the sum of all elements in the matrix.
        /// </summary>
        /// <returns>Sum of all elements</returns>
        float Sum();

        /// <summary>
        /// Initializes all elements with random values.
        /// </summary>
        /// <param name="seed">Random seed for reproducibility</param>
        /// <param name="min">Minimum random value (inclusive)</param>
        /// <param name="max">Maximum random value (inclusive)</param>
        void SetRandom(int seed, float min, float max);

        /// <summary>
        /// Sets the diagonal elements to the specified value.
        /// </summary>
        /// <param name="diagonalValue">Value to set on diagonal</param>
        void SetDiagonal(float diagonalValue);

        // Convolution operations

        /// <summary>
        /// Performs 2D convolution with the given kernel.
        /// </summary>
        /// <param name="kernel">Convolution kernel/filter</param>
        /// <returns>Convolution result matrix</returns>
        IMatrix Convolution(IMatrix kernel);

        /// <summary>
        /// Performs full 2D convolution with the given kernel.
        /// </summary>
        /// <param name="kernel">Convolution kernel/filter</param>
        /// <returns>Full convolution result matrix</returns>
        IMatrix ConvolutionFull(IMatrix kernel);

        /// <summary>
        /// Computes the Hadamard (element-wise) product with another matrix.
        /// </summary>
        /// <param name="other">Matrix to multiply element-wise</param>
        /// <returns>New matrix with element-wise product</returns>
        IMatrix HadamardProduct(IMatrix other);

        public static IMatrix operator *(IMatrix lhs, float scalar) => lhs.Multiply(scalar);
        public static IMatrix operator *(float scalar, IMatrix lhs) => lhs.Multiply(scalar);

    }
}