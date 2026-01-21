namespace MatrixLib
{
    public class AnnTensor : Tensor
    {
        public AnnTensor(AvxMatrix matrix, AvxColumnVector columnVector)
        {
            Matrix = matrix;
            ColumnVector = columnVector;
        }
        public AvxMatrix Matrix { get; }
        public override List<AvxMatrix> Matrices { get { return new List<AvxMatrix>() { Matrix }; } }
        public AvxColumnVector ColumnVector { get; }

        public override Tensor Add(Tensor rhs)
        {
            throw new NotImplementedException();
        }

        public override Tensor Multiply(Tensor rhs_)
        {
            AnnTensor rhs = rhs_ as AnnTensor;
            if (rhs.ColumnVector != null & this.ColumnVector != null)
                return (this.ColumnVector * rhs.ColumnVector).ToTensor();
            else
                return (this.Matrix * rhs.Matrix).ToTensor();
        }
    }

}
