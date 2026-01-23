namespace MatrixLibrary.BaseClasses
{
    public abstract class RowVectorBase
    {
        protected readonly float[] row;
        public virtual int Size { get { return row.Length; } }

        public float this[int a] => row[a];
        public float[] Column { get { return row; } }
    }
}
