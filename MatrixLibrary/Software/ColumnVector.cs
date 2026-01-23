using System.Diagnostics;
using MatrixLibrary.BaseClasses;

namespace MatrixLibrary
{
    public class ColumnVector : ColumnVectorBase
    {
        public MatrixBackend Backend => MatrixBackend.Software;

        public ColumnVector(float[] inputVector) : base(inputVector) { }
        public ColumnVector(int size) : base(size) { }
        public override float Sum()
        {
            float accum = 0;
            for (int r = 0; r < Size; r++)
            {
                accum += this.Column[r];
            }
            return accum;
        }

        public override ColumnVector Log()
        {
            ColumnVector vector = new ColumnVector(this.Size);
            for (int i = 0; i < this.Size; i++)
            {
				column[i] =	(float)Math.Log(this[i]);
            }
            return vector;
        }
        public void SetRandom(int seed, float min, float max)
        {
            Random rnd = new Random(seed);
            float width = max - min;
            for (int c = 0; c < this.Size; c++)
            {
                column[c] = (float)((rnd.NextDouble() * width) + min);
                // for testing only:  Mat[r, c] = 0.5;
            }
        }

        public override MatrixBase OuterProduct(ColumnVectorBase right)
        {
            return this.OuterProduct(right.Column);
        }

        private MatrixBase OuterProduct(float[] right)
        {
            Matrix2D result = new Matrix2D(this.Size, right.Length);
            for (int r = 0; r < this.Size; r++)
            {
                for (int c = 0; c < right.Length; c++)
                {
                    result[r, c] = this[r] * right[c];
                }
            }
            return result;
        }

        public override ColumnVector Subtract(ColumnVectorBase right)
        {
            float[] res = new float[this.Size];
            for (int i = 0; i < this.Size; i++)
            {
                res[i] = this[i] - right[i];
            }

            return new ColumnVector(res);
        }

        public override ColumnVector Add(ColumnVectorBase right)
        {
            Debug.Assert(this.Size == right.Size);
            float[] res = new float[this.Size];
            for (int i = 0; i < this.Size; i++)
            {
                res[i] = this[i] + right[i];
            }

            return new ColumnVector(res);
        }

        public override ColumnVector Multiply(ColumnVectorBase right)
        {
            float[] res = new float[this.Size];
            for (int i = 0; i < this.Size; i++)
            {
                res[i] = this[i] * right[i];
            }
            return new ColumnVector(res);
        }

        public override ColumnVector Multiply(float scalar)
        {
            float[] res = new float[this.Size];
            for (int i = 0; i < this.Size; i++)
            {
                res[i] = this[i] * scalar;
            }
            return new ColumnVector(res);
        }
        public Matrix2D Multiply(RowVector rowVector)
        {
            Matrix2D result = new Matrix2D(this.Size, rowVector.Size);
            for(int c = 0; c < this.Size; c++)
                for(int r = 0; r < rowVector.Size; r++)
                    result[c, r] = this[c] * rowVector[r];
            return result;
        }

        public override ColumnVector Add(float scalar)
        {
            float[] res = new float[this.Size];
            for (int i = 0; i < this.Size; i++)
            {
                res[i] = this[i] + scalar;
            }
            return new ColumnVector(res);
        }

        public override ColumnVector Subtract(float scalar)
        {
            float[] res = new float[this.Size];
            for (int i = 0; i < this.Size; i++)
            {
                res[i] = scalar - this[i];
            }

            return new ColumnVector(res);
        }

        public override MatrixBase RhsOuterProduct(Tensor lhs)
        {
            throw new NotImplementedException();
        }

        public override MatrixBase OuterProduct(FlattenedMatricesAsVector rhs)
        {
            throw new NotImplementedException();
        }
    }
}
