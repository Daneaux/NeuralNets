
using MatrixLib.Interfaces;

namespace MatrixLib.Software
{
    public class RowVector : IRowVector
    {
        public float[] Row { get; private set; }
        public int Size => Row.Length;

        //public MatrixBackend Backend => MatrixBackend.Software;
        /*        public override float this[int r, int c]
                {
                    get => Row[r];
                    set => Row[r] = value;
                }*/

        public RowVector(float[] inputVector)
        {
            this.Row = inputVector;
        }

        public RowVector(int size)
        {
            this.Row = new float[size];
        }

        public float this[int i]
        {
            get { return this.Row[i]; }
            set { this.Row[i] = value; }
        }

        // IRowVector interface methods
        public void SetRandom(int seed, int min, int max)
        {
            Random rnd = new(seed);
            float width = max - min;
            for (int c = 0; c < Size; c++)
            {
                this[c] = (float)((rnd.NextDouble() * width) + min);
            }
        }

        float IRowVector.Sum()
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
