
namespace MatrixLibrary.BaseClasses
{
    // 
    // Matrix2D and ColumnVector and RowVector (all software only matrix/vec classes) are only here 
    // for benchmarking purposes to see how much faster AVX can be. Nothing else uses these classes.
    // I suppose once I implement the GPU accelerated matrix math, I'll get rid of these
    // 

    public abstract class MatrixBase
    {
        public float[,] Mat { get; protected set; }

        public int Rows { get; protected set; }
        public int Cols { get; protected set; }

        public int TotalSize => Rows * Cols;
        public bool IsSquare() => Rows == Cols;

        public float this[int r, int c]
        {
            get { return this.Mat[r, c]; }
            set { this.Mat[r, c] = value; }
        }
        public void SetRandom(int seed, float min, float max)
        {
            Random rnd = new Random(seed);
            float width = max - min;
            for (int c = 0; c < Cols; c++)
            {
                for (int r = 0; r < Rows; r++)
                {
                    this[r, c] = (float)((rnd.NextDouble() * width) + min);
                    // for testing only:  Mat[r, c] = 0.5;
                }
            }
        }
        public virtual float Sum()
        {
            float sum = 0;
            for (int i = 0; i < this.Rows; ++i)
            {
                for (int j = 0; j < this.Cols; ++j)
                {
                    sum += this[i, j];
                }
            }
            return sum;
        }
        public abstract MatrixBase Log();
        public abstract void SetDiagonal(float diagonalValue);

        public abstract MatrixBase Add(float scalar);
        public abstract MatrixBase Add(MatrixBase other);
        public static MatrixBase operator +(MatrixBase lhs, MatrixBase rhs) => lhs.Add(rhs);
        public static MatrixBase operator +(MatrixBase lhs, float scalar) => lhs.Add(scalar);

        public abstract MatrixBase Subtract(MatrixBase other);
        public static MatrixBase operator -(MatrixBase lhs, MatrixBase rhs) => lhs.Subtract(rhs);

        public abstract MatrixBase Multiply(float scalar);
        public static MatrixBase operator *(float scalar, MatrixBase lhs) => lhs.Multiply(scalar);
        public static MatrixBase operator *(MatrixBase lhs, float scalar) => lhs.Multiply(scalar);

        public abstract MatrixBase Multiply(MatrixBase other);
        public static MatrixBase operator *(MatrixBase lhs, MatrixBase rhs) => lhs.Multiply(rhs);
        public abstract ColumnVectorBase MatrixTimesColumn(ColumnVectorBase column);
        public static ColumnVectorBase operator *(MatrixBase lhs, ColumnVectorBase rhs) => lhs.MatrixTimesColumn(rhs);

        public abstract MatrixBase Convolution(MatrixBase kernel);
        public abstract MatrixBase ConvolutionFull(MatrixBase kernel);

        public abstract MatrixBase HadamardProduct(MatrixBase other);

        //public abstract RowVectorBase RowTimesMatrix(RowVectorBase left);
        //public static RowVectorBase operator *(RowVectorBase lhs, MatrixBase rhs) => rhs.RowTimesMatrix(lhs);


        // TODO BUG: this seems redundant, what was the intent?
        public abstract MatrixBase Transpose();
        public abstract MatrixBase GetTransposedMatrix();
    }
}
