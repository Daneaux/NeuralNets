
namespace MatrixLib.Interfaces
{

    /// <summary>
    /// Interface for row vector operations supporting multiple backend implementations (Software, AVX, GPU).
    /// </summary>
    public interface IRowVector
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
        float[] Row { get; }

        /// <summary>
        /// Backend implementation type.
        /// </summary>
        //MatrixBackend Backend { get; }

        // Utility operations

        /// <summary>
        /// Computes the sum of all elements in the vector.
        /// </summary>
        /// <returns>Sum of all elements</returns>
        float Sum();
    }
}
