using MatrixLibrary.BaseClasses;

namespace MatrixLibrary
{
    public class RowVector : RowVectorBase
    {
        public float[] Row { get; private set; }
        public int Size { get { return this.Row.Length; } }
        public MatrixBackend Backend => MatrixBackend.Software;

        public float this[int r]
        {
            get => Row[r];
            set => Row[r] = value;
        }

        public RowVector(float[] inputVector)
        {
            this.Row = inputVector;
        }

        public RowVector(int size)
        {
            this.Row = new float[size];
        }

        public float Sum()
        {
            float sum = 0;
            for (int i = 0; i < Size; i++)
            {
                sum += this.Row[i];
            }
            return sum;
        }
    }
}
