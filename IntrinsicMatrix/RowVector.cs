using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MatrixLibrary
{    public class RowVector : Matrix_Base
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
}
