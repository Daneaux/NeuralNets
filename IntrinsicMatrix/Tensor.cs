using System;
using System.Collections.Generic;
using System.Linq;
using System.Net.NetworkInformation;
using System.Text;
using System.Threading.Tasks;
using static TorchSharp.torch.nn;

namespace MatrixLibrary
{   
    public static class TensorExtensions
    {
        public static Tensor ToTensor(this ColumnVector input)
        {
            return new AnnTensor(null, new AvxColumnVector(input.Column));
        }
        public static Tensor ToTensor(this Matrix2D input)
        {
            return new AnnTensor(new AvxMatrix(input.Mat), null);
        }

        public static Tensor ToTensor(this AvxMatrix matrix)
        {
            return new AnnTensor(matrix, null);
        }

        public static Tensor ToTensor(this AvxColumnVector columnVector)
        {
            return new AnnTensor(null, columnVector);
        }
    }

    public abstract class Tensor
    {
        public static Tensor operator +(Tensor lhs, Tensor rhs) => lhs.Add(rhs);
        public abstract Tensor Add(Tensor rhs);
    }

    public class ConvolutionTensor : Tensor
    {
        public ConvolutionTensor(List<AvxMatrix> matrices)
        {
            Matrices = matrices;
        }
        public List<AvxMatrix> Matrices { get; private set; }

        public override Tensor Add(Tensor rhs)
        {
            throw new NotImplementedException();
        }
    }

    public class AnnTensor : Tensor
    {
        public AnnTensor(AvxMatrix matrix, AvxColumnVector columnVector)
        {
            Matrix = matrix;
            ColumnVector = columnVector;
        }
        public AvxMatrix Matrix { get; }
        public AvxColumnVector ColumnVector { get; }

        public override Tensor Add(Tensor rhs)
        {
            throw new NotImplementedException();
        }
    }

}
