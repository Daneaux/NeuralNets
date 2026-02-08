using MatrixLibrary.BaseClasses;

namespace MatrixLibrary
{
    /// <summary>
    /// GPU row vector — not supported. Use GpuColumnVector instead.
    /// Matches the AVX backend pattern where RowVector is unsupported.
    /// </summary>
    public class GpuRowVector : RowVectorBase
    {
        public GpuRowVector(int size)
        {
            throw new NotSupportedException("GPU backend does not support RowVector. Use ColumnVector instead.");
        }

        public GpuRowVector(float[] data)
        {
            throw new NotSupportedException("GPU backend does not support RowVector. Use ColumnVector instead.");
        }
    }
}
