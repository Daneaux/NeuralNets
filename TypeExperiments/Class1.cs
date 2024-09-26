using System.Numerics;

namespace TypeExperiments
{
    // What manner of magic is this?
    public interface IMatrixMan<IMat, IVec>
        where IMat : IAdditionOperators<IMat, IVec, IVec>
        //where IVec : IAdditionOperators<IVec, IVec, IVec>

    {
        static abstract IVec operator +(IMatrixMan<IMat, IVec> lhs, IVec rhs);
    }

    public class SuperVec : IMatrixMan<AVector2, AVector>
    {
        static AVector IMatrixMan<AVector2, AVector>.operator +(IMatrixMan<AVector2, AVector> lhs, AVector rhs)
        {
            throw new NotImplementedException();
        }
    }

    public class FooBarContext<WT, BT>
        where WT : IAdditionOperators<WT, BT, WT>, IMultiplyOperators<WT, WT, WT>
    {
        public WT Operate(WT ww, BT bb)
        {
            WT foo = ww + bb;
            return foo;
        }

        public WT Op2(WT ww, WT w2)
        {
            WT foo = ww * w2;
            return foo;
        }
    }

    public class AVector2 : IAdditionOperators<AVector2, AVector, AVector>
    {
        public static AVector operator +(AVector2 left, AVector right)
        {
            throw new NotImplementedException();
        }

        private Matrix2 Stuff(Matrix2 rhs)
        {
            throw new NotImplementedException();
        }
    }


    public class AVector : IAdditionOperators<AVector, AVector2, AVector2>
    {
        public static Matrix2 operator +(AVector lhs, Matrix2 rhs) => lhs.Stuff(rhs);

        public static AVector2 operator +(AVector left, AVector2 right)
        {
            throw new NotImplementedException();
        }

        private Matrix2 Stuff(Matrix2 rhs)
        {
            throw new NotImplementedException();
        }
    }

}
