using MatrixLib.Interfaces;
using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Text;

namespace MatrixLib.Software
{
    public class Matrix2D : IMatrix
    {
        public int Rows { get; private set; }
        public int Cols { get; private set; }
        public int TotalSize => Rows * Cols;
        //public MatrixBackend Backend => MatrixBackend.Software;
        public float[,] Mat { get; private set; }

        public Matrix2D(int r, int c)
        {
            Rows = r;
            Cols = c;
            Mat = new float[Rows, Cols];
        }

        public Matrix2D(float[,] m)
        {
            Rows = m.GetLength(0);
            Cols = m.GetLength(1);
            this.Mat = m; // no deep copy, better not change my matrix dude!
        }

        public unsafe Matrix2D Convolution(Matrix2D filter)
        {
            // slide a 4x4 filter across this matrix.
            // resulting matrix dimensions are: lhx - filter.x + 1 

            Debug.Assert(filter != null);
            Debug.Assert(filter.Rows < this.Rows);
            Debug.Assert(filter.Cols < this.Cols);

            (int rows, int cols) = Matrix2D.ConvolutionSizeHelper(this, filter);
            Matrix2D result = new Matrix2D(rows, cols);

            int stride = this.Cols;

            for (int t = 0; t < rows; t++)
            {
                for (int l = 0; l < cols; l++)
                {
                    int srcx = l;
                    int srcy = t;

                    for (int r = 0; r < filter.Rows; r++, srcy++)
                    {
                        srcx = l;
                        for (int c = 0; c < filter.Cols; c++, srcx++)
                        {
                            result[t, l] += filter[r, c] * this[srcy, srcx];
                        }
                    }
                }
            }

            return result;
        }

        public unsafe Matrix2D ConvolutionFull(Matrix2D filter)
        {
            // Full convolution - output size is: (matrix.x + filter.x - 1, matrix.y + filter.y - 1)
            Debug.Assert(filter != null);

            int outputRows = this.Rows + filter.Rows - 1;
            int outputCols = this.Cols + filter.Cols - 1;
            Matrix2D result = new Matrix2D(outputRows, outputCols);

            for (int outRow = 0; outRow < outputRows; outRow++)
            {
                for (int outCol = 0; outCol < outputCols; outCol++)
                {
                    int srcx = outCol;
                    int srcy = outRow;

                    float sum = 0;
                    for (int r = 0; r < filter.Rows; r++)
                    {
                        srcy = outRow + r;
                        for (int c = 0; c < filter.Cols; c++)
                        {
                            srcx = outCol + c;
                            // Flip kernel for full convolution
                            sum += this[srcy, srcx] * filter[filter.Rows - 1 - r, filter.Cols - 1 - c];
                        }
                    }
                    result[outRow, outCol] = sum;
                }
            }

            return result;
        }

        public static (int r, int c) ConvolutionSizeHelper(Matrix2D matrix, Matrix2D filter)
        {
            int cols = matrix.Cols - filter.Cols + 1;
            int rows = matrix.Rows - filter.Rows + 1;
            return (rows, cols);
        }

        public virtual Matrix2D Log()
        {
            Matrix2D logMat = new Matrix2D(this.Cols, this.Rows);
            for (int i = 0; i < this.Rows; ++i)
            {
                for (int j = 0; j < this.Cols; ++j)
                {
                    logMat[i, j] = (float)Math.Log(this[i, j]);
                }
            }
            return logMat;
        }

        public void SetDiagonal(float diagonalValue)
        {
            Debug.Assert(Rows == Cols);
            for (int i = 0; i < Rows; i++)
                Mat[i, i] = diagonalValue;
        }

        public static Matrix2D operator +(Matrix2D a, Matrix2D b) => a.Add(b);

        private Matrix2D Add(Matrix2D b)
        {
            if (this.Rows != b.Rows || this.Cols != b.Cols)
            {
                throw new ArgumentException("bad dimensions in Matrix.add");
            }

            Matrix2D res = new Matrix2D(Rows, Cols);
            for (int r = 0; r < Rows; r++)
            {
                for (int c = 0; c < Cols; c++)
                {
                    res.Mat[r, c] = this[r, c] + b[r, c];
                }
            }

            return res;
        }

        // I'm on the left of 'm'
        public virtual Matrix2D Multiply(Matrix2D m)
        {
            if (this.Cols == m.Rows)
            {
                Matrix2D res = new Matrix2D(this.Rows, m.Cols);
                for (int r = 0; r < Rows; r++)
                {
                    // multiply my horizontal vector times m's vertical vector
                    // my r and it's C
                    int leftR = r;
                    for (int rightCol = 0; rightCol < m.Cols; rightCol++)
                    {
                        res.Mat[leftR, rightCol] = DoRowTimesColumn(leftR, rightCol, m);
                    }
                }

                return res;
            }
            else
            {
                throw new ArgumentOutOfRangeException("Bad dimensions");
            }
        }

        public static RowVector operator *(RowVector left, Matrix2D right) => right.RowTimesMatrix(left);

        private RowVector RowTimesMatrix(RowVector left)
        {
            if (left.Size == this.Rows)
            {
                float[] vector = new float[left.Size];
                for (int c = 0; c < this.Cols; c++)
                {
                    vector[c] = DoRowVectorTimesColumn(left, c);
                }
                return new RowVector(vector);
            }
            else
            {
                throw new ArgumentOutOfRangeException("Bad dimensions");
            }
        }

        public static ColumnVector operator *(Matrix2D left, ColumnVector right) => left.MatrixTimesColumn(right);

        public ColumnVector MatrixTimesColumn(ColumnVector colVec)
        {
            if (this.Cols == colVec.Size)
            {
                float[] vector = new float[this.Rows];
                for (int r = 0; r < Rows; r++)
                {
                    vector[r] = DoRowTimesColumnVector(r, colVec);
                }

                return new ColumnVector(vector);
            }
            else
            {
                throw new ArgumentOutOfRangeException("Bad dimensions");
            }
        }

        public static Matrix2D operator *(Matrix2D a, Matrix2D b) => a.Multiply(b);

        public float this[int r, int c]
        {
            get { return this.Mat[r, c]; }
            set { this.Mat[r, c] = value; }
        }

        public static Matrix2D operator *(float scalar, Matrix2D b) => b.Multiply(scalar);
        public static Matrix2D operator *(Matrix2D b, float scalar) => b.Multiply(scalar);

        public Matrix2D Multiply(float scalar)
        {
            Matrix2D res = new Matrix2D(Rows, Cols);
            for (int r = 0; r < Rows; r++)
            {
                for (int c = 0; c < Cols; c++)
                {
                    res.Mat[r, c] = scalar * this[r, c];
                }
            }
            return res;
        }

        public static Matrix2D operator -(Matrix2D a, Matrix2D b) => a.Subtract(b);

        private Matrix2D Subtract(Matrix2D b)
        {
            Debug.Assert(HasSameDimensions(b));

            // this minus b
            if (this.Cols == b.Cols && this.Rows == b.Rows)
            {
                Matrix2D res = new Matrix2D(this.Rows, b.Cols);
                for (int r = 0; r < Rows; r++)
                {
                    for (int c = 0; c < this.Cols; c++)
                    {
                        res[r, c] = this[r, c] - b[r, c];
                    }
                }
                return res;
            }
            else
            {
                throw new ArgumentOutOfRangeException("Bad dimensions");
            }
        }

        public Matrix2D HadamardProduct(Matrix2D b)
        {
            if (this.HasSameDimensions(b))
            {
                Matrix2D res = new Matrix2D(Rows, Cols);
                for (int r = 0; r < Rows; r++)
                {
                    for (int c = 0; c < this.Cols; c++)
                    {
                        res.Mat[r, c] = this.Mat[r, c] * b.Mat[r, c];
                    }
                }
                return res;
            }
            return null;
        }

        private bool HasSameDimensions(Matrix2D b) => (Rows == b.Rows) && (Cols == b.Cols);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private float DoRowTimesColumn(int myRow, int rightMatrixCol, Matrix2D rightMatrix)
        {
            float cum = 0;
            for (int i = 0; i < Cols; i++)
            {
                cum += this[myRow, i] * rightMatrix[i, rightMatrixCol];
            }
            return cum;
        }

        private float DoRowTimesColumnVector(int myRow, ColumnVector colVec)
        {
            Debug.Assert(colVec.Size == this.Cols);
            float product = 0;
            for (int c = 0; c < Cols; c++)
            {
                product += this.Mat[myRow, c] * colVec[c];
            }
            return product;
        }

        private float DoRowVectorTimesColumn(RowVector rowVec, int myCol)
        {
            Debug.Assert(rowVec.Size == this.Rows);
            float cum = 0;
            for (int i = 0; i < rowVec.Size; i++)
            {
                cum += rowVec[i] * this.Mat[i, myCol];
            }
            return cum;
        }

        public Matrix2D GetTransposedMatrix()
        {
            Matrix2D mt = new Matrix2D(this.Cols, this.Rows);
            for (int c = 0; c < this.Cols; c++)
            {
                for (int r = 0; r < this.Rows; r++)
                {
                    mt.Mat[c, r] = this.Mat[r, c];
                }
            }
            return mt;
        }

        public void Print()
        {
            StringBuilder str = new StringBuilder();
            for (int i = 0; i < this.Rows; ++i)
            {
                for (int j = 0; j < this.Cols; ++j)
                {
                    str.Append(this.Mat[i, j].ToString("F3").PadLeft(8) + " ");
                }
                str.AppendLine();
            }
            Console.Write(str);
        }

        // IMatrix interface methods

        IMatrix IMatrix.Add(IMatrix other)
        {
            if (other is Matrix2D otherMatrix)
                return this.Add(otherMatrix);
            throw new NotSupportedException("Add operation with non-Matrix2D backend not yet implemented");
        }

        IMatrix IMatrix.Subtract(IMatrix other)
        {
            if (other is Matrix2D otherMatrix)
                return this.Subtract(otherMatrix);
            throw new NotSupportedException("Subtract operation with non-Matrix2D backend not yet implemented");
        }

        IMatrix IMatrix.Multiply(IMatrix other)
        {
            if (other is Matrix2D otherMatrix)
                return this.Multiply(otherMatrix);
            throw new NotSupportedException("Multiply operation with non-Matrix2D backend not yet implemented");
        }

        IMatrix IMatrix.Multiply(float scalar)
        {
            return this.Multiply(scalar);
        }

        IColumnVector IMatrix.Multiply(IColumnVector vector)
        {
            if (vector is ColumnVector colVec)
                return this.MatrixTimesColumn(colVec);
            throw new NotSupportedException("Multiply with non-ColumnVector backend not yet implemented");
        }

        IMatrix IMatrix.Transpose()
        {
            return this.GetTransposedMatrix();
        }

        IMatrix IMatrix.Log()
        {
            return this.Log();
        }

        IMatrix IMatrix.Convolution(IMatrix kernel)
        {
            if (kernel is Matrix2D kernelMatrix)
                return this.Convolution(kernelMatrix);
            throw new NotSupportedException("Convolution with non-Matrix2D kernel not yet implemented");
        }

        IMatrix IMatrix.ConvolutionFull(IMatrix kernel)
        {
            if (kernel is Matrix2D kernelMatrix)
                return this.ConvolutionFull(kernelMatrix);
            throw new NotSupportedException("ConvolutionFull with non-Matrix2D kernel not yet implemented");
        }

        IMatrix IMatrix.HadamardProduct(IMatrix other)
        {
            if (other is Matrix2D otherMatrix)
                return this.HadamardProduct(otherMatrix);
            throw new NotSupportedException("HadamardProduct with non-Matrix2D backend not yet implemented");
        }

        void IMatrix.SetRandom(int seed, float min, float max)
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

        public float Sum()
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
    }
}
