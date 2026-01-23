using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace TypeExperiments
{

    // this approach skips interfaces (like IMatrix and IVector or IColumnVector). Seems better overall.
    public class Collector
    {
        List<FooBase> fooBar;
        Collector(List<FooBase> foo)
        {
            this.fooBar = foo;
        }
    }

    public abstract class BaseVec
    {
        public float[] Vec { get; protected set; }
        protected BaseVec() { }

        public abstract BaseVec Add(BaseVec a);
        public static BaseVec operator +(BaseVec a, BaseVec b) => a.Add(b);
    }

    public class SoftVec : BaseVec
    {
        public override BaseVec Add(BaseVec a)
        {
            throw new NotImplementedException();
        }
    }


    public abstract class FooBase
    {
        float[,] FooBar;
        protected FooBase() { }
        FooBase(float[,] FooBar) { }

        public abstract FooBase Add(FooBase b);
        public abstract FooBase Add(float scalar);
        public static FooBase operator +(FooBase a, FooBase b) => a.Add(b);
        public static FooBase operator +(FooBase a, float scalar) => a.Add(scalar);

        public abstract FooBase Add(BaseVec vec);
        public static FooBase operator +(FooBase a, BaseVec vec) => a.Add(vec);
    }

    public class ExampleMatrix : FooBase
    {
        public ExampleMatrix() { }

        public override ExampleMatrix Add(FooBase a)
        {
            return new ExampleMatrix();
        }

        public override FooBase Add(float scalar)
        {
            return new ExampleMatrix();
        }

        public override FooBase Add(BaseVec vec)
        {
            for (int i = 0; i < 10; i++)
            {
                vec.Vec[i] = i;
            }
            return this;
        }
    }


}
