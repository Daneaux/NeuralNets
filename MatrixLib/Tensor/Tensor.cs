
namespace MatrixLib
{
    public abstract class Tensor
    {
        public static Tensor operator +(Tensor lhs, Tensor rhs) => lhs.Add(rhs);
        public abstract Tensor Add(Tensor rhs);

        public static Tensor operator *(Tensor lhs, Tensor rhs) => lhs.Multiply(rhs);
        public abstract Tensor Multiply(Tensor rhs);

        public abstract List<AvxMatrix> Matrices { get; }
    }

}
