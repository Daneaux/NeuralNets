using BenchmarkDotNet.Reports;
using IntrinsicMatrix;
using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Text;

namespace NeuralNets
{

    public static class Extension
    {
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

    public abstract class Matrix_Base
    {
        public int Rows { get; protected set; }
        public int Cols { get; protected set; }

        public abstract float this[int r, int c]
        {
            get;
            set;
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
    }

    public class Matrix2D : Matrix_Base
    {
        public float[,] Mat { get; private set; }

        public Matrix2D(int r, int c)
        {
            Rows = r;
            Cols = c;
            Mat = new float[Rows, Cols];
        }

/*        protected Matrix2D(int rows, int cols, bool singleD)
        {
            Debug.Assert(singleD == true);  // so lame
            Rows = rows;
            Cols = cols;
            Mat = null;
        }*/

        /*        public Matrix(float[] inputVector) : this(inputVector.Length, 1)
                {
                    for (int r = 0; r < Rows; r++)
                    {
                        Mat[r, 0] = inputVector[r];
                    }
                }*/

/*        public Matrix2D(ColumnVector[] columns, int numRows) : this(numRows, columns.Length)
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
        }*/

        public Matrix2D(float[,] m)
        {
            Rows = m.GetLength(0);
            Cols = m.GetLength(1);
            this.Mat = m; // no deep copy, better not change my matrix dude!
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
        public override float this[int r, int c]
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
                    for (int c = 0; c < Cols; c++)
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
            for(int c =0; c < Cols; c++)
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
    }

    /*
     * [ a, c, d, e, f, ... z ]
     */
    public class RowVector : Matrix_Base
    {
        public float[] Row { get; private set; }
        public int Size { get { return this.Cols; } }

        public override float this[int r, int c]
        {
            get => Row[r];
            set => Row[r] = value;
        }

        public RowVector(float[] inputVector) 
        {
            this.Row = inputVector;
            this.Cols = inputVector.Length;
            this.Rows = 1;
        }

        public RowVector(int size) 
        {
            this.Row = new float[size];
            this.Cols = size;
            this.Rows = 1;
        }

        public float this[int i]
        {
            get { return this.Row[i]; }
            set { this.Row[i] = value; }
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
    public class ColumnVector : Matrix_Base
    {        
        public int Size { get { return this.Rows; } }

        public float[] Column { get; private set ; }

        public ColumnVector(float[] inputVector)
        {
            this.Cols = 1;
            this.Rows = inputVector.Length;
            this.Column = inputVector;
        }
        public ColumnVector(int size)
        { 
            this.Column = new float[size];
            this.Cols = 1;
            this.Rows = this.Column.Length;
        }
        public float this[int i]
        {
            get { return this.Column[i]; }
            set { this.Column[i] = value; }
        }
        public override float this[int r, int c]
        {
            get => Column[c];
            set => Column[c] = value;
        }

        public override float Sum()
        {
            float accum = 0;
            for(int r = 0; r < Rows; r++)
            {
                accum += this.Column[r];
            }
            return accum;
        }

        public static Matrix2D operator *(ColumnVector left, RowVector right) => left.OuterProduct(right);

        private Matrix2D OuterProduct(RowVector right)
        {
            Matrix2D result = new Matrix2D(this.Size, right.Size);
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
            float[] res = new float[this.Size];
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
            float[] res = new float[this.Size];
            for (int i = 0; i < this.Size; i++)
            {
                res[i] = this[i] + right[i];
            }

            return new ColumnVector(res);
        }

        public static ColumnVector operator *(ColumnVector left, ColumnVector right) => left.Multiply(right);

        public ColumnVector Multiply(ColumnVector right)
        {
            float[] res = new float[this.Size];
            for(int i=0; i < this.Size; i++)
            {
                res[i] = this[i] * right[i];
            }
            return new ColumnVector(res);
        }

        public static ColumnVector operator *(float scalar, ColumnVector vec) => vec.ScalarMultiply(scalar);
        public static ColumnVector operator *(ColumnVector vec, float scalar) => vec.ScalarMultiply(scalar);

        private  ColumnVector ScalarMultiply(float scalar)
        {
            float[] res = new float[this.Size];
            for (int i = 0; i < this.Size; i++)
            {
                res[i] = this[i] * scalar;
            }

            return new ColumnVector(res);
        }

        public static ColumnVector operator +(ColumnVector vec, float scalar) => vec.ScalarAddition(scalar);
        public static ColumnVector operator +(float scalar, ColumnVector vec) => vec.ScalarAddition(scalar);

        private ColumnVector ScalarAddition(float scalar)
        {
            float[] res = new float[this.Size];
            for (int i = 0; i < this.Size; i++)
            {
                res[i] = this[i] + scalar;
            }

            return new ColumnVector(res);
        }

        public static ColumnVector operator -(ColumnVector vec, float scalar) => vec.ScalarAddition(-scalar);
        public static ColumnVector operator -(float scalar, ColumnVector vec) => vec.ScalarSubtract(scalar);

        private ColumnVector ScalarSubtract(float scalar)
        {
            float[] res = new float[this.Size];
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

        public float GetMax()
        {
            float max = float.MinValue;
            for (int i = 0; i < this.Size; i++)
            {
                max = Math.Max(max, this[i]);
            }
            return max;
        }

        public float SumExpEMinusMax(float max)
        {
            float scale = 0;
            for (int i = 0; i < this.Size; i++)
            {
               scale += (float)Math.Exp(this[i] - max);
            }
            return scale;
        }

        // todo: column vector not really supposed to know how to generate softmax ...
        public ColumnVector SoftmaxHelper()
        {
            float max = this.GetMax();
            float scaleFactor = this.SumExpEMinusMax(max);
            ColumnVector softMaxVector = new ColumnVector(this.Size);
            for (int i = 0; i < this.Size; i++)
            {
                softMaxVector[i] = (float)(Math.Exp(this[i] - max) / scaleFactor);
            }
            return softMaxVector;
        }

        public ColumnVector Log()
        {
            ColumnVector vector = new ColumnVector(this.Size);
            for(int i = 0;i < this.Size;i++)
            {
                vector[i] = (float)Math.Log(this[i]);
            }
            return vector;
        }
    }

} // namespace

