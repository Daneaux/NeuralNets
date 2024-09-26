using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using System.Text;
using System.Threading.Tasks;

namespace MatrixLibrary
{
    public interface IMatrix
    {
        public int Rows { get; }
        public int Cols { get; }

        public float[,] Mat { get; }
        public float this[int r, int c] { get; }
        public static abstract IMatrix operator +(IMatrix a, IMatrix b);
        public static abstract IMatrix operator -(IMatrix a, IMatrix b);
        
        public static abstract IMatrix operator *(IMatrix a, IMatrix b);
        public static abstract IMatrix operator *(float scalar, IMatrix b);
        public static abstract IMatrix operator *(IMatrix a, float scalar);

        public static abstract IRowVector operator *(IRowVector left, IMatrix right);

        public static abstract IColumnVector operator *(IMatrix left, IColumnVector right);
        public float Sum();
        public IMatrix Log();
    }

    public abstract class BMatrix
    {
        public int Rows { get; protected set; }
        public int Cols { get; protected set; }

        public virtual float[,] Mat { get; protected set; }
        public abstract float this[int r, int c] { get; }
        public static BMatrix operator +(BMatrix a, BMatrix b) => b.Add(a);
        public static BMatrix operator -(BMatrix a, BMatrix b) { return null; }

        public abstract BMatrix Add(BMatrix b);
/*
        public static abstract BMatrix operator *(IMatrix a, IMatrix b);
        public static abstract BMatrix operator *(float scalar, IMatrix b);
        public static abstract BMatrix operator *(IMatrix a, float scalar);

        public static abstract IRowVector operator *(IRowVector left, IMatrix right);

        public static abstract IColumnVector operator *(IMatrix left, IColumnVector right);
        public float Sum();
        public IMatrix Log();*/
    }



}
