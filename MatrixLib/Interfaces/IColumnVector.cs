namespace MatrixLib.Interfaces
{
    /// <summary>
    /// Interface for column vector operations supporting multiple backend implementations (Software, AVX, GPU).
    /// </summary>
    public interface IColumnVector
    {
        /// <summary>
        /// Number of elements in the vector.
        /// </summary>
        int Size { get; }

        /// <summary>
        /// Gets or sets the value at the specified index.
        /// </summary>
        /// <param name="i">Index (0-based)</param>
        /// <returns>Value at specified index</returns>
        float this[int i] { get; set; }

        /// <summary>
        /// Gets the underlying data array (for compatibility with existing code).
        /// </summary>
        float[] Column { get; }

        /// <summary>
        /// Backend implementation type.
        /// </summary>
       // MatrixBackend Backend { get; }

        // Arithmetic operations

        /// <summary>
        /// Adds another column vector to this vector element-wise.
        /// </summary>
        /// <param name="other">Vector to add</param>
        /// <returns>New vector containing the sum</returns>
        IColumnVector Add(IColumnVector other);
        public static IColumnVector operator +(IColumnVector left, IColumnVector right) => left.Add(right);

        /// <summary>
        /// Subtracts another column vector from this vector element-wise.
        /// </summary>
        /// <param name="other">Vector to subtract</param>
        /// <returns>New vector containing the difference</returns>
        IColumnVector Subtract(IColumnVector other);
        public static IColumnVector operator -(IColumnVector lhs, IColumnVector rhs) => lhs.Subtract(rhs);

        /// <summary>
        /// Multiplies this vector by another vector element-wise.
        /// </summary>
        /// <param name="other">Vector to multiply by</param>
        /// <returns>New vector with element-wise product</returns>
        IColumnVector Multiply(IColumnVector other);
        public static IColumnVector operator *(IColumnVector left, IColumnVector right) => left.Multiply(right);
        /// <summary>
        /// Multiplies this vector by a scalar.
        /// </summary>
        /// <param name="scalar">Scalar value to multiply by</param>
        /// <returns>New vector with all elements multiplied by scalar</returns>
        IColumnVector Multiply(float scalar);
        public static IColumnVector operator *(IColumnVector left, float scalar) => left.Multiply(scalar);

        /// <summary>
        /// Adds a scalar to this vector element-wise.
        /// </summary>
        /// <param name="scalar">Scalar value to add</param>
        /// <returns>New vector with scalar added to each element</returns>
        IColumnVector Add(float scalar);
        public static IColumnVector operator +(IColumnVector left, float scalar) => left.Add(scalar);

        /// <summary>
        /// Subtracts a scalar from this vector element-wise.
        /// </summary>
        /// <param name="scalar">Scalar value to subtract</param>
        /// <returns>New vector with scalar subtracted from each element</returns>
        IColumnVector Subtract(float scalar);
        public static IColumnVector operator -(IColumnVector lhs, float scalar) => lhs.Subtract(scalar);
        // Utility operations

        /// <summary>
        /// Computes the natural logarithm of each element.
        /// </summary>
        /// <returns>New vector with log of each element</returns>
        IColumnVector Log();

        /// <summary>
        /// Computes the sum of all elements in the vector.
        /// </summary>
        /// <returns>Sum of all elements</returns>
        float Sum();

        /// <summary>
        /// Finds the maximum value in the vector.
        /// </summary>
        /// <returns>Maximum value</returns>
        float GetMax();

        /// <summary>
        /// Computes the outer product with another column vector.
        /// </summary>
        /// <param name="other">Vector to compute outer product with</param>
        /// <returns>Matrix representing outer product</returns>
        IMatrix OuterProduct(IColumnVector other);

        /// <summary>
        /// Initializes all elements with random values.
        /// </summary>
        /// <param name="seed">Random seed for reproducibility</param>
        /// <param name="min">Minimum random value (inclusive)</param>
        /// <param name="max">Maximum random value (inclusive)</param>
        void SetRandom(int seed, int min, int max);
    }
}
