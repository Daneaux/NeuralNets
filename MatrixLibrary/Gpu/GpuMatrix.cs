
using MatrixLibrary.BaseClasses;

namespace MatrixLibrary
{
    /// <summary>
    /// GPU-accelerated matrix implementation (stub - throws NotImplementedException).
    /// This stub allows the factory to create GPU instances without requiring
    /// a full GPU implementation. In Phase 3, these methods will be
    /// replaced with actual CUDA/OpenCL implementations.
    /// </summary>
    public class GpuMatrix : MatrixBase
    {
        public GpuMatrix(int rows, int cols)
        {
            Rows = rows;
            Cols = cols;
            Mat = new float[rows, cols];
        }

        public GpuMatrix(float[,] data)
        {
            Mat = data;
            Rows = data.GetLength(0);
            Cols = data.GetLength(1);
        }

        public override MatrixBase Log()
        {
            throw new NotImplementedException();
        }

        public override void SetDiagonal(float diagonalValue)
        {
            throw new NotImplementedException();
        }

        public override MatrixBase Add(float scalar)
        {
            throw new NotImplementedException();
        }

        public override MatrixBase Add(MatrixBase other)
        {
            throw new NotImplementedException();
        }

        public override MatrixBase Subtract(MatrixBase other)
        {
            throw new NotImplementedException();
        }

        public override MatrixBase Multiply(float scalar)
        {
            throw new NotImplementedException();
        }

        public override MatrixBase Multiply(MatrixBase other)
        {
            throw new NotImplementedException();
        }

        public override ColumnVectorBase MatrixTimesColumn(ColumnVectorBase column)
        {
            throw new NotImplementedException();
        }

        public override MatrixBase Convolution(MatrixBase kernel)
        {
            throw new NotImplementedException();
        }

        public override MatrixBase ConvolutionFull(MatrixBase kernel)
        {
            throw new NotImplementedException();
        }

        public override MatrixBase HadamardProduct(MatrixBase other)
        {
            throw new NotImplementedException();
        }

        public override MatrixBase Transpose()
        {
            throw new NotImplementedException();
        }

        public override MatrixBase GetTransposedMatrix()
        {
            throw new NotImplementedException();
        }
    }
}
