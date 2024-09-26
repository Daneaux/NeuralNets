using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MatrixLibrary
{
    public interface IColumnVector
    {
        public int Size { get; }
        public float[] Column { get; }
        public float this[int i] { get; }

        public float Sum();
        public float GetMax();
        public IColumnVector Log();


        // outer product explicit
        public IMatrix OuterProduct(float[] vector);
        public static abstract IMatrix operator *(IColumnVector left, IRowVector right);

        public static abstract IColumnVector operator *(IColumnVector left, float scalar);
        public static abstract IColumnVector operator *(float scalar, IColumnVector right);

        public static abstract IColumnVector operator +(IColumnVector left, float scalar);
        public static abstract IColumnVector operator +(float scalar, IColumnVector right);

        public static abstract IColumnVector operator -(IColumnVector left, float scalar);
        public static abstract IColumnVector operator -(float scalar, IColumnVector right);

        public static abstract IColumnVector operator +(IColumnVector left, IColumnVector right);
        public static abstract IColumnVector operator -(IColumnVector left, IColumnVector right);
        public static abstract IColumnVector operator *(IColumnVector left, IColumnVector right);
    }
}
