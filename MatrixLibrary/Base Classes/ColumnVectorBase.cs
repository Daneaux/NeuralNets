
namespace MatrixLibrary.BaseClasses
{
    public abstract class ColumnVectorBase
    {
        protected readonly float[] column;
        public virtual int Size { get { return column.Length; } }

        public float this[int a] => column[a];
        public float[] Column { get { return column; } }

        public ColumnVectorBase() { }
        public ColumnVectorBase(float[] column) { this.column = column; }
        public ColumnVectorBase(int size) { this.column = new float[size]; }
        public float GetMax()
        {
            float max = float.MinValue;
            for (int i = 0; i < column.Length; i++)
            {
                max = Math.Max(max, this[i]);
            }
            return max;
        }
        public void SetRandom(int seed, int min, int max)
        {
            Random rnd = new Random(seed);
            float width = max - min;
            for (int r = 0; r < Size; r++)
            {
                column[r] = (float)((rnd.NextDouble() * width) + min);
            }
        }

        public abstract float Sum();
        public abstract ColumnVectorBase Log();

        public abstract ColumnVectorBase Multiply(float scalar);
        public static ColumnVectorBase operator *(ColumnVectorBase vec, float scalar) => vec.Multiply(scalar);
        public static ColumnVectorBase operator *(float scalar, ColumnVectorBase vec) => vec.Multiply(scalar);

        public abstract ColumnVectorBase Multiply(ColumnVectorBase rhs);
        public static ColumnVectorBase operator *(ColumnVectorBase lhs, ColumnVectorBase rhs) => lhs.Multiply(rhs);

        //public abstract ColumnVectorBase MatrixTimesColumn(MatrixBase lhs);
        //public static ColumnVectorBase operator *(MatrixBase lhs, ColumnVectorBase rhs) => rhs.MatrixTimesColumn(lhs);

        public abstract ColumnVectorBase Add(float scalar);
        public abstract ColumnVectorBase Add(ColumnVectorBase rhs);
        public static ColumnVectorBase operator +(ColumnVectorBase vec, float scalar) => vec.Add(scalar);
        public static ColumnVectorBase operator +(float scalar, ColumnVectorBase vec) => vec.Add(scalar);
        public static ColumnVectorBase operator -(ColumnVectorBase vec, float scalar) => vec.Add(-scalar);
        public static ColumnVectorBase operator +(ColumnVectorBase lhs, ColumnVectorBase rhs) => lhs.Add(rhs);

        public abstract ColumnVectorBase Subtract(float scalar);
        public abstract ColumnVectorBase Subtract(ColumnVectorBase rhs);
        public static ColumnVectorBase operator -(float scalar, ColumnVectorBase vec) => vec.Subtract(scalar);
        public static ColumnVectorBase operator -(ColumnVectorBase lhs, ColumnVectorBase rhs) => lhs.Subtract(rhs);


        public abstract MatrixBase RhsOuterProduct(Tensor lhs);
        public abstract MatrixBase OuterProduct(ColumnVectorBase rhs);
        public abstract MatrixBase OuterProduct(FlattenedMatricesAsVector rhs);
    }
}
