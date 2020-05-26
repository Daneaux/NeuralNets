using System;
using System.CodeDom;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

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
            for(int r = 0; r < Rows; r++)
            {
                Mat[r, 0] = inputVector[r];
            }
        }

        public Matrix (double [,] m)
        {
            Rows = m.GetLength(0);
            Cols = m.GetLength(1);
            this.Mat = m; // no deep copy, better not change my matrix dude!
        }

        public void SetRandom(int seed, double min, double max)
        {
            Random rnd = new Random(seed);
            double width = max - min;
            for(int c = 0; c < Cols; c++)
            {
                for(int r = 0; r < Rows; r++)
                {
                    Mat[r, c] = (rnd.NextDouble() * width) + min;
                }
            }
        }

        // I'm on the left of 'm'
        public Matrix Multiply(Matrix m)
        {
            if(this.Cols == m.Rows)
            {
                Matrix res = new Matrix(this.Rows, m.Cols);
                for (int r = 0; r < Rows; r++)
                {
                    // multiply my horizontal vector times m's vertical vector
                    // my r and it's C
                    int leftR = r;
                    for (int rightCol = 0; rightCol < m.Cols; rightCol++)
                    {
                        res.Mat[leftR, rightCol] = DoVectorMultLeft(leftR, rightCol, m);
                    }
                }

                return res;
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

        public Matrix HadamardProduct(Matrix b)
        {
            if(this.HasSameDimensions(b))
            {
                Matrix res = new Matrix(Rows, Cols);
                for(int r=0; r<Rows; r++)
                {
                    for(int c=0; c<Cols; c++)
                    {
                        res.Mat[r, c] = this.Mat[r, c] * b.Mat[r, c];
                    }
                }
                return res;
            }
            return null;
        }

        private bool HasSameDimensions(Matrix b) => (Rows == b.Rows) && (Cols == b.Cols);

        private double DoVectorMultLeft(int leftR, int rightCol, Matrix m)
        {
            double cum = 0;
            for(int i = 0; i < Cols; i++)
            {
                cum += this.Mat[leftR, i] * m.Mat[i, rightCol];
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
            for(int i = 0; i < d; i++)
            {
                this.Mat[i, i] = 1;
            }
        }
    }
}
