using MatrixLib.Interfaces;
using MatrixLib.Software;
using System.Diagnostics;

namespace MatrixLib
{
    public class ColumnVector : IColumnVector
    {
        public int Size { get; private set; }
        public float[] Column { get; private set; }
        public ColumnVector(float[] inputVector)
        {
            this.Column = inputVector;
            this.Size = inputVector.Length;
        }
        public ColumnVector(int size)
        {
            this.Size = size;
            this.Column = new float[size];
        }
        public float this[int i]
        {
            get { return this.Column[i]; }
            set { this.Column[i] = value; }
        }

        public float Sum()
        {
            float accum = 0;
            for (int r = 0; r < Size; r++)
            {
                accum += this.Column[r];
            }
            return accum;
        }

        public static Matrix2D operator *(ColumnVector left, RowVector right) => left.OuterProduct(right);

        public Matrix2D OuterProduct(ColumnVector lastActivation)
        {
            return this.OuterProduct(new RowVector(lastActivation.Column));
        }

        public Matrix2D OuterProduct(RowVector right)
        {
            Matrix2D result = new Matrix2D(this.Size, right.Size);
            for (int r = 0; r < this.Size; r++)
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
            for (int i = 0; i < this.Size; i++)
            {
                res[i] = this[i] - right[i];
            }

            return new ColumnVector(res);
        }

        public static ColumnVector operator +(ColumnVector left, ColumnVector right) => left.PlusColumnVector(right);

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
            for (int i = 0; i < this.Size; i++)
            {
                res[i] = this[i] * right[i];
            }
            return new ColumnVector(res);
        }

        public static ColumnVector operator *(float scalar, ColumnVector vec) => vec.ScalarMultiply(scalar);
        public static ColumnVector operator *(ColumnVector vec, float scalar) => vec.ScalarMultiply(scalar);

        private ColumnVector ScalarMultiply(float scalar)
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

/*        public RowVector Transpose()
        {
            RowVector result = new RowVector(this.Size);
            for (int i = 0; i < this.Size; i++)
            {
                result[i] = this[i];
            }
            return result;
        }*/

        public float GetMax()
        {
            float max = float.MinValue;
            for (int i = 0; i < this.Size; i++)
            {
                max = Math.Max(max, this[i]);
            }
            return max;
        }

        public ColumnVector Log()
        {
            ColumnVector vector = new ColumnVector(this.Size);
            for (int i = 0; i < this.Size; i++)
            {
                vector[i] = (float)Math.Log(this[i]);
            }
            return vector;
        }

        // IColumnVector interface methods

        IColumnVector IColumnVector.Add(IColumnVector other)
        {
            if (other is ColumnVector otherVector)
                return this.PlusColumnVector(otherVector);
            throw new NotSupportedException("Add operation with non-ColumnVector backend not yet implemented");
        }

        IColumnVector IColumnVector.Subtract(IColumnVector other)
        {
            if (other is ColumnVector otherVector)
                return this.MinusColumnVector(otherVector);
            throw new NotSupportedException("Subtract operation with non-ColumnVector backend not yet implemented");
        }

        IColumnVector IColumnVector.Multiply(IColumnVector other)
        {
            if (other is ColumnVector otherVector)
                return this.Multiply(otherVector);
            throw new NotSupportedException("Multiply operation with non-ColumnVector backend not yet implemented");
        }

        IColumnVector IColumnVector.Multiply(float scalar)
        {
            return this.ScalarMultiply(scalar);
        }

        IColumnVector IColumnVector.Add(float scalar)
        {
            return this.ScalarAddition(scalar);
        }

        IColumnVector IColumnVector.Subtract(float scalar)
        {
            return this.ScalarSubtract(scalar);
        }

        IColumnVector IColumnVector.Log()
        {
            return this.Log();
        }

        float IColumnVector.GetMax()
        {
            return this.GetMax();
        }

        IMatrix IColumnVector.OuterProduct(IColumnVector other)
        {
            if (other is ColumnVector otherVector)
                return this.OuterProduct(otherVector);
            throw new NotSupportedException("OuterProduct with non-ColumnVector backend not yet implemented");
        }

        public void SetRandom(int seed, int min, int max)
        {
            Random rnd = new(seed);
            float width = max - min;
            for (int c = 0; c < Size; c++)
            {
                this[c] = (float)((rnd.NextDouble() * width) + min);
            }
        }

    }
}
