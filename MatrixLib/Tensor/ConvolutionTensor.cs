using System.Diagnostics;

namespace MatrixLib
{
    public class ConvolutionTensor : Tensor
    {
        public ConvolutionTensor(List<AvxMatrix> matrices)
        {
            Matrices = matrices;
        }
        public override List<AvxMatrix> Matrices { get; }

        public override Tensor Add(Tensor rhs)
        {
            throw new NotImplementedException();
        }
        public override Tensor Multiply(Tensor rhs)
        {
            Debug.Assert(this.Matrices.Count == rhs.Matrices.Count);
            List<AvxMatrix> result = new List<AvxMatrix>();
            for (int i = 0; i < Matrices.Count; i++)
            {
                result.Add(Matrices[i] * rhs.Matrices[i]);
            }
            return result.ToTensor();
        }
    }

}
