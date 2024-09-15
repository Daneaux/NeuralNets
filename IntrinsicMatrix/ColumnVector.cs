using System.Diagnostics;

namespace MatrixLibrary
{
    public class ColumnVector : Matrix_Base
    {
        public int Size { get; private set; }

        public float[] Column { get; private set; }
        public override float this[int r, int c]
        {
            get => Column[c];
            set => Column[c] = value;
        }

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

        public override float Sum()
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
            for (int i = 0; i < this.Size; i++)
            {
                vector[i] = (float)Math.Log(this[i]);
            }
            return vector;
        }
    }
}
