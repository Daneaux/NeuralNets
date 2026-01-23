using MatrixLibrary.BaseClasses;
using System.Diagnostics;

namespace MatrixLibrary
{
    public static class MatrixHelpers
    {
        public static ColumnVectorBase UnrollMatricesToColumnVector(List<MatrixBase> matrices)
        {
            int size = matrices.Count * matrices[0].TotalSize;
            float[] floats = new float[size];
            int i = 0;
            foreach (MatrixBase mat in matrices)
            {
                for (int r = 0; r < mat.Rows; r++)
                    for (int c = 0; c < mat.Cols; c++)
                        floats[i++] = mat[r, c];
            }

            ColumnVectorBase result = MatrixFactory.CreateColumnVector(floats);
            return result;
        }


        public static (int r, int c) ConvolutionSizeHelper(MatrixBase matrix, MatrixBase filter)
        {
            int cols = matrix.Cols - filter.Cols + 1;
            int rows = matrix.Rows - filter.Rows + 1;
            return (rows, cols);
        }

        public static (int r, int c) ConvolutionSizeHelper(MatrixBase matrix, int kernelSize, bool isFull = false, int stride = 1)
        {
            // W = input volume
            // K = kernel size
            // P = padding (not used here yet)
            // S = stride
            // result size = 1 + (W - K + 2P) / S
            //  filter is square
            int cols = 1 + ((matrix.Cols - kernelSize) / stride);
            int rows = 1 + ((matrix.Rows - kernelSize) / stride);

            if (isFull)
            {
                cols += 2 * (kernelSize - 1);
                rows += 2 * (kernelSize - 1);
            }

#if DEBUG
            if (isFull && stride == 1)
            {
                Debug.Assert(cols == matrix.Cols + kernelSize - 1);
                Debug.Assert(rows == matrix.Rows + kernelSize - 1);
            }
#endif

            return (rows, cols);
        }
        public static (int r, int c) ConvolutionSizeHelper(InputOutputShape inputShape, int kernelSize, bool isFull = false, int stride = 1)
        {
            // W = input volume
            // K = kernel size
            // P = padding (not used here yet)
            // S = stride
            // result size = 1 + (W - K + 2P) / S
            //  filter is square
            int cols = 1 + ((inputShape.Width - kernelSize) / stride);
            int rows = 1 + ((inputShape.Height - kernelSize) / stride);

            if (isFull)
            {
                cols += 2 * (kernelSize - 1);
                rows += 2 * (kernelSize - 1);
            }

            return (rows, cols);
        }
    }
}