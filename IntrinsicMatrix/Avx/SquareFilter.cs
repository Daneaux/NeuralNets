namespace MatrixLibrary.Avx
{
    public class SquareKernel : AvxMatrix
    {
        public int FilterSize { get; }
        public SquareKernel(float[,] mat) : base(mat)
        {
            if (mat.GetLength(0) != mat.GetLength(1))
            {
                throw new ArgumentException("filter matrix needs to be square");
            }
            FilterSize = mat.GetLength(0);
        }

        public SquareKernel(int filterSize) : base(filterSize, filterSize)
        {
            FilterSize = filterSize;
        }

        public static SquareKernel operator +(SquareKernel lhs, SquareKernel rhs)
        {
            AvxMatrix ret = lhs.AddMatrix(rhs);
            return new SquareKernel(ret.Mat);
        }
        public static SquareKernel operator -(SquareKernel lhs, SquareKernel rhs)
        {
            AvxMatrix ret = lhs.SubtractMatrix(rhs);
            return new SquareKernel(ret.Mat);
        }

        public static SquareKernel operator *(SquareKernel lhs, float scalar) => lhs.MultiplyScalarSquareKernel(scalar);
        public static SquareKernel operator *(float scalar, SquareKernel lhs) => lhs.MultiplyScalarSquareKernel(scalar);

        public SquareKernel MultiplyScalarSquareKernel(float scalar)
        {
            AvxMatrix ret = this.MultiplyScalar(scalar);
            return new SquareKernel(ret.Mat);
        }

    }
}
