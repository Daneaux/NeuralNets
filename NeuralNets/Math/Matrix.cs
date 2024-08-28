using System.Diagnostics;
using System.Net.NetworkInformation;
using System.Text;

namespace NeuralNets
{
    public class Matrix
    {
        public double[,] Mat { get; private set; }
        public int Rows { get; private set; }
        public int Cols { get; private set; }
        public Matrix(int r, int c)
        {
            Rows = r;
            Cols = c;
            Mat = new double[Rows, Cols];
        }

        public Matrix(double[] inputVector) : this(inputVector.Length, 1)
        {
            for (int r = 0; r < Rows; r++)
            {
                Mat[r, 0] = inputVector[r];
            }
        }

        public Matrix(ColumnVector[] columns, int numRows) : this(numRows, columns.Length)
        {
            int c = 0;
            foreach(ColumnVector col in columns)
            {
                for(int r= 0; r<col.Size; r++)
                {
                    this[r, c] = col[r];
                }
                c++;
            }
        }

        public Matrix(double[,] m)
        {
            Rows = m.GetLength(0);
            Cols = m.GetLength(1);
            this.Mat = m; // no deep copy, better not change my matrix dude!
        }

        public void SetRandom(int seed, double min, double max)
        {
            Random rnd = new Random(seed);
            double width = max - min;
            for (int c = 0; c < Cols; c++)
            {
                for (int r = 0; r < Rows; r++)
                {
                   Mat[r, c] = (rnd.NextDouble() * width) + min;
                   // for testing only:  Mat[r, c] = 0.5;
                }
            }
        }

        public static Matrix operator +(Matrix a, Matrix b) => a.Add(b);

        private Matrix Add(Matrix b)
        {
            if (this.Rows != b.Rows || this.Cols != b.Cols)
            {
                throw new ArgumentException("bad dimensions in Matrix.add");
            }

            Matrix res = new Matrix(Rows, Cols);
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
        public virtual Matrix Multiply(Matrix m)
        {
            if (this.Cols == m.Rows)
            {
                Matrix res = new Matrix(this.Rows, m.Cols);
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

        public static RowVector operator *(RowVector left, Matrix right) => right.RowTimesMatrix(left);

        private RowVector RowTimesMatrix(RowVector left)
        {
            if (left.Size == this.Rows)
            {
                double[] vector = new double[left.Size];
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

        public static ColumnVector operator *(Matrix left, ColumnVector right) => left.MatrixTimesColumn(right);

        public ColumnVector MatrixTimesColumn(ColumnVector colVec)
        {
            if (this.Cols == colVec.Size)
            {
                double[] vector = new double[this.Rows];
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

        public static Matrix operator *(Matrix a, Matrix b) => a.Multiply(b);
        public double this[int r, int c]
        {
            get { return this.Mat[r, c]; }
            set { this.Mat[r, c] = value; }
        }

        public static Matrix operator *(double scalar, Matrix b) => b.Multiply(scalar);
        public static Matrix operator *(Matrix b, double scalar) => b.Multiply(scalar);
        public Matrix Multiply(double scalar)
        {
            Matrix res = new Matrix(Rows, Cols);
            for (int r = 0; r < Rows; r++)
            {
                for (int c = 0; c < Cols; c++)
                {
                    res.Mat[r, c] = scalar * this[r, c];
                }
            }
            return res;
        }

        public static Matrix operator -(Matrix a, Matrix b) => a.Subtract(b);

        private Matrix Subtract(Matrix b)
        {
            Debug.Assert(HasSameDimensions(b));

            // this minus b
            if (this.Cols == b.Cols && this.Rows == b.Rows)
            {
                Matrix res = new Matrix(this.Rows, b.Cols);
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

        public Matrix HadamardProduct(Matrix b)
        {
            if (this.HasSameDimensions(b))
            {
                Matrix res = new Matrix(Rows, Cols);
                for (int r = 0; r < Rows; r++)
                {
                    for (int c = 0; c < Cols; c++)
                    {
                        res.Mat[r, c] = this.Mat[r, c] * b.Mat[r, c];
                    }
                }
                return res;
            }
            return null;
        }

        private bool HasSameDimensions(Matrix b) => (Rows == b.Rows) && (Cols == b.Cols);

        private double DoRowTimesColumn(int myRow, int rightMatrixCol, Matrix rightMatrix)
        {
            double cum = 0;
            for (int i = 0; i < Cols; i++)
            {
                cum += this[myRow, i] * rightMatrix[i, rightMatrixCol];
            }
            return cum;
        }

        private double DoRowTimesColumnVector(int myRow, ColumnVector colVec)
        {
            Debug.Assert(colVec.Size == this.Cols);
            double product = 0;
            for(int c =0; c < Cols; c++)
            {
                product += this.Mat[myRow, c] * colVec[c];
            }
            return product;
        }

        private double DoRowVectorTimesColumn(RowVector rowVec, int myCol)
        {
            Debug.Assert(rowVec.Size == this.Rows);
            double cum = 0;
            for (int i = 0; i < rowVec.Size; i++)
            {
                cum += rowVec[i] * this.Mat[i, myCol];
            }
            return cum;
        }

        public Matrix GetTransposedMatrix()
        {
            Matrix mt = new Matrix(this.Cols, this.Rows);
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

        public virtual Matrix Log()
        {
            Matrix logMat = new Matrix(this.Cols, this.Rows);
            for (int i = 0; i < this.Rows; ++i)
            {
                for (int j = 0; j < this.Cols; ++j)
                {
                    logMat[i, j] = Math.Log(this.Mat[i, j]);
                }
            }
            return logMat;
        }

        public virtual double Sum()
        {
            double sum = 0;
            for (int i = 0; i < this.Rows; ++i)
            {
                for (int j = 0; j < this.Cols; ++j)
                {
                    sum += this.Mat[i, j];
                }
            }
            return sum;
        }
    }

    /*
     * [ a, c, d, e, f, ... z ]
     */
    public class RowVector : Matrix
    {
        public int Size { get { return this.Cols; } }
        public RowVector(double[] inputVector) : base(1, inputVector.Length)
        {
            for (int c = 0; c < Size; c++)
            {
                Mat[0, c] = inputVector[c];
            }
        }

        public RowVector(int size) :  base(1, size) { }

        public double this[int i]
        {
            get { return this.Mat[0, i]; }
            set { this.Mat[0, i] = value; }
        }
    }
    
    /* 
     * --
     * a
     * b
     * c
     * d
     * ..
     * y
     * z
     * --
     *
     */
    public class ColumnVector : Matrix
    {
        public int Size { get { return this.Rows; } }
        public ColumnVector(double[] inputVector) : base(inputVector.Length, 1)
        {
            for (int r = 0; r < Rows; r++)
            {
                Mat[r, 0] = inputVector[r];
            }
        }

        public ColumnVector(int size) : base(size, 1) { }

        public override double Sum()
        {
            double accum = 0.0;
            for(int r = 0; r < Rows; r++)
            {
                accum += Mat[r, 0];
            }
            return accum;
        }

        public static Matrix operator *(ColumnVector left, RowVector right) => left.OuterProduct(right);

        private Matrix OuterProduct(RowVector right)
        {
            Matrix result = new Matrix(this.Size, right.Size);
            for(int r = 0; r < this.Size; r++)
            {
                for (int c = 0; c < right.Size; c++)
                {
                    result[r, c] = this[r] * right[c];
                }
            }
            return result;
        }

        public static ColumnVector operator -(ColumnVector left, ColumnVector right)
        {
            return left.MinusColumnVector(right);
        }

        private ColumnVector MinusColumnVector(ColumnVector right)
        {
            double[] res = new double[this.Size];
            for(int i = 0; i < this.Size; i++)
            {
                res[i] = this[i] - right[i];
            }

            return new ColumnVector(res);
        }

        public static ColumnVector operator +(ColumnVector left, ColumnVector right)
        {
            return left.PlusColumnVector(right);
        }

        private ColumnVector PlusColumnVector(ColumnVector right)
        {
            Debug.Assert(this.Size == right.Size);
            double[] res = new double[this.Size];
            for (int i = 0; i < this.Size; i++)
            {
                res[i] = this[i] + right[i];
            }

            return new ColumnVector(res);
        }

        public static ColumnVector operator *(ColumnVector left, ColumnVector right) => left.Multiply(right);

        public ColumnVector Multiply(ColumnVector right)
        {
            double[] res = new double[this.Size];
            for(int i=0; i < this.Size; i++)
            {
                res[i] = this[i] * right[i];
            }
            return new ColumnVector(res);
        }

        public static ColumnVector operator *(double scalar, ColumnVector vec) => vec.ScalarMultiply(scalar);
        public static ColumnVector operator *(ColumnVector vec, double scalar) => vec.ScalarMultiply(scalar);

        private  ColumnVector ScalarMultiply(double scalar)
        {
            double[] res = new double[this.Size];
            for (int i = 0; i < this.Size; i++)
            {
                res[i] = this[i] * scalar;
            }

            return new ColumnVector(res);
        }

        public static ColumnVector operator +(ColumnVector vec, double scalar) => vec.ScalarAddition(scalar);
        public static ColumnVector operator +(double scalar, ColumnVector vec) => vec.ScalarAddition(scalar);

        private ColumnVector ScalarAddition(double scalar)
        {
            double[] res = new double[this.Size];
            for (int i = 0; i < this.Size; i++)
            {
                res[i] = this[i] + scalar;
            }

            return new ColumnVector(res);
        }

        public static ColumnVector operator -(ColumnVector vec, double scalar) => vec.ScalarAddition(-scalar);
        public static ColumnVector operator -(double scalar, ColumnVector vec) => vec.ScalarSubtract(scalar);

        private ColumnVector ScalarSubtract(double scalar)
        {
            double[] res = new double[this.Size];
            for (int i = 0; i < this.Size; i++)
            {
                res[i] = scalar - this[i];
            }

            return new ColumnVector(res);
        }

        public RowVector Transpose()
        {
            RowVector result = new RowVector(this.Size);
            for(int i = 0;i < this.Size;i++)
            {
                result[i] = this[i];
            }
            return result;
        }

        public double GetMax()
        {
            double max = double.MinValue;
            for (int i = 0; i < this.Size; i++)
            {
                max = Math.Max(max, this[i]);
            }
            return max;
        }

        public double SumExpEMinusMax(double max)
        {
            double scale = 0;
            for (int i = 0; i < this.Size; i++)
            {
                scale += Math.Exp(this[i] - max);
            }
            return scale;
        }

        // todo: column vector not really supposed to know how to generate softmax ...
        public ColumnVector SoftmaxHelper()
        {
            double max = this.GetMax();
            double scaleFactor = this.SumExpEMinusMax(max);
            ColumnVector softMaxVector = new ColumnVector(this.Size);
            for (int i = 0; i < this.Size; i++)
            {
                softMaxVector[i] = Math.Exp(this[i] - max) / scaleFactor;
            }
            return softMaxVector;
        }

        public override ColumnVector Log()
        {
            ColumnVector vector = new ColumnVector(this.Size);
            for(int i = 0;i < this.Size;i++)
            {
                vector[i] = Math.Log(this[i]);
            }
            return vector;
        }

        public double this[int i]
        {
            get { return this.Mat[i, 0]; }
            set { this.Mat[i, 0] = value; }
        }
    }

    public class SquareMatrix : Matrix
    {
        public SquareMatrix(int d) : base(d, d)
        {
        }

        public Matrix GetInvertedMatrix()
        {
            return null;
        }

        public double Determinant()
        {
            return 0;
        }
    }

    public class IdendityMatrix : SquareMatrix
    {
        public IdendityMatrix(int d) : base(d)
        {
            for (int i = 0; i < d; i++)
            {
                this.Mat[i, i] = 1;
            }
        }
    }

} // namespace

