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
    }
}
