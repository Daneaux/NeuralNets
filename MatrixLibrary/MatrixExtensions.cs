
using MatrixLibrary.BaseClasses;

namespace MatrixLibrary
{
    public static class MatrixExtensions
    {

        public static void XavierInitialize(this MatrixBase matrix, int fanIn, int fanOut, int seed)
        {
            float bound = (float)Math.Sqrt(6.0 / (fanIn + fanOut));
            matrix.SetRandom(seed, -bound, bound);
        }

        public static void XavierInitialize(this ColumnVectorBase columnVector, int fanIn, int fanOut, int seed)
        {
            float bound = (float)Math.Sqrt(6.0 / (fanIn + fanOut));
            columnVector.SetRandom(seed, -bound, bound);
        }

        public static AvxColumnVector ToAvxVector(this ColumnVector columnVector)
        {
            return new AvxColumnVector(columnVector.Column);
        }

        public static ColumnVector ToColumnVector(this AvxColumnVector avxColumnVector)
        {
            return new ColumnVector(avxColumnVector.Column);
        }

        public static Matrix2D ToMatrix2d(this AvxMatrix matrix)
        {
            return new Matrix2D(matrix.Mat);
        }

        public static AvxMatrix ToAvxMatrix(this Matrix2D matrix)
        {
            return new AvxMatrix(matrix.Mat);            
        }
    }
}


