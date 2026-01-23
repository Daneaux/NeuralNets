using System.Diagnostics;
using MatrixLibrary.BaseClasses;

namespace MatrixLibrary
{
    public static class TensorExtensions
    {
        public static Tensor ToTensor(this List<MatrixBase> input)
        {
            return new ConvolutionTensor(input);
        }
        public static Tensor ToTensor(this ColumnVectorBase input)
        {
            return new AnnTensor(null, new AvxColumnVector(input.Column));
        }
        public static Tensor ToTensor(this Matrix2D input)
        {
            return new AnnTensor(new AvxMatrix(input.Mat), null);
        }

        public static Tensor ToTensor(this MatrixBase matrix)
        {
            return new AnnTensor(matrix, null);
        }

        public static ColumnVectorBase? ToColumnVector(this Tensor tensor)
        {
            return (tensor as AnnTensor)?.ColumnVector;
        }
        public static MatrixBase? ToMatrix(this Tensor tensor)
        {
            return (tensor as AnnTensor)?.Matrix;
        }

        public static FlattenedMatricesAsVector? ToFlattenedMatrices(this Tensor tensor)
        {
            List<MatrixBase> mats = tensor.Matrices;
            return new FlattenedMatricesAsVector(mats);
        }
    }

    public abstract class Tensor
    {
        public static Tensor operator +(Tensor lhs, Tensor rhs) => lhs.Add(rhs);
        public abstract Tensor Add(Tensor rhs);

        public static Tensor operator *(Tensor lhs, Tensor rhs) => lhs.Multiply(rhs);
        public abstract Tensor Multiply(Tensor rhs);

        public abstract List<MatrixBase> Matrices { get; }
    }

    public class ConvolutionTensor : Tensor
    {
        public ConvolutionTensor(List<MatrixBase> matrices)
        {
            Matrices = matrices;
        }
        public override List<MatrixBase> Matrices { get; }

        public override Tensor Add(Tensor rhs)
        {
            throw new NotImplementedException();
        }
        public override Tensor Multiply(Tensor rhs)
        {
            Debug.Assert(this.Matrices.Count == rhs.Matrices.Count);
            List<MatrixBase> result = new List<MatrixBase>();
            for(int i = 0; i < Matrices.Count; i++)
            {
                result.Add(Matrices[i].Multiply(rhs.Matrices[i]));
            }
            return result.ToTensor();
        }
    }

    public class AnnTensor : Tensor
    {
        public AnnTensor(MatrixBase matrix, ColumnVectorBase columnVector)
        {
            Matrix = matrix;
            ColumnVector = columnVector;
        }
        public MatrixBase Matrix { get; }
        public override List<MatrixBase> Matrices { get { return new List<MatrixBase>() { Matrix }; } }
        public ColumnVectorBase ColumnVector { get; }

        public override Tensor Add(Tensor rhs)
        {
            throw new NotImplementedException();
        }

        public override Tensor Multiply(Tensor rhs_)
        {
            AnnTensor rhs = rhs_ as AnnTensor;
            if (rhs.ColumnVector != null & this.ColumnVector != null)
                return (ColumnVector * rhs.ColumnVector).ToTensor();
            else
                return (Matrix * rhs.Matrix).ToTensor();
        }
    }

}
