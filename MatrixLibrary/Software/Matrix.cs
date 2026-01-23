using System.Text;
using System.Diagnostics;
using MatrixLibrary.BaseClasses;
using System.Runtime.CompilerServices;

namespace MatrixLibrary
{

    public class Matrix2D : MatrixBase
    {
        public MatrixBackend Backend => MatrixBackend.Software;

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

            (int rows, int cols) = MatrixHelpers.ConvolutionSizeHelper(this, filter);
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

        public override Matrix2D Log()
        {
            Matrix2D logMat = new Matrix2D(this.Rows, this.Cols);
            for (int r = 0; r < this.Rows; ++r)
            {
                for (int c = 0; c < this.Cols; ++c)
                {
                    logMat[r, c] = (float)Math.Log(this[r, c]);
                }
            }
            return logMat;
        }

        public override void SetDiagonal(float diagonalValue)
        {
            Debug.Assert(Rows == Cols);
            for (int i = 0; i < Rows; i++)
                Mat[i, i] = diagonalValue;
        }

        public override Matrix2D Add(MatrixBase b)
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
        public override Matrix2D Multiply(MatrixBase m)
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

/*        public RowVector RowTimesMatrix(RowVector left)
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
*/

        public override ColumnVector MatrixTimesColumn(ColumnVectorBase colVec)
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

        public override Matrix2D Multiply(float scalar)
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

        public override Matrix2D Subtract(MatrixBase b)
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

        public override Matrix2D HadamardProduct(MatrixBase b)
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

        private bool HasSameDimensions(MatrixBase b) => (Rows == b.Rows) && (Cols == b.Cols);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private float DoRowTimesColumn(int myRow, int rightMatrixCol, MatrixBase rightMatrix)
        {
            float result = 0;
            for (int i = 0; i < Cols; i++)
                result += this[myRow, i] * rightMatrix[i, rightMatrixCol];
            return result;
        }

        private float DoRowTimesColumnVector(int myRow, ColumnVectorBase colVec)
        {
            Debug.Assert(colVec.Size == this.Cols);
            float product = 0;
            for(int c = 0; c < Cols; c++)
            {
                product += this.Mat[myRow, c] * colVec[c];
            }
            return product;
        }

        private float DoRowVectorTimesColumn(RowVectorBase rowVec, int myCol)
        {
            Debug.Assert(rowVec.Size == this.Rows);
            float cum = 0;
            for (int i = 0; i < rowVec.Size; i++)
            {
                cum += rowVec[i] * this.Mat[i, myCol];
            }
            return cum;
        }

        public override Matrix2D GetTransposedMatrix()
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

        public override MatrixBase Add(float scalar)
        {
            Matrix2D res = new Matrix2D(Rows, Cols);
            for (int r = 0; r < Rows; r++)
                for (int c = 0; c < Cols; c++)
                    res.Mat[r, c] = this[r, c] + scalar;

            return res;
        }

        public override MatrixBase Convolution(MatrixBase kernel)
        {
            throw new NotImplementedException();
        }

        public override MatrixBase ConvolutionFull(MatrixBase kernel)
        {
            throw new NotImplementedException();
        }

        public override MatrixBase Transpose()
        {
            throw new NotImplementedException();
        }
    }
}
