using MatrixLib.Software;

namespace MatrixLib
{
    public static class TensorExtensions
    {
        public static Tensor ToTensor(this List<AvxMatrix> input)
        {
            return new ConvolutionTensor(input);
        }
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

        public static AvxColumnVector? ToAvxColumnVector(this Tensor tensor)
        {
            return (tensor as AnnTensor)?.ColumnVector;
        }
        public static AvxMatrix? ToAvxMatrix(this Tensor tensor)
        {
            return (tensor as AnnTensor)?.Matrix;
        }

        public static FlattenedMatricesAsVector? ToFlattenedMatrices(this Tensor tensor)
        {
            List<AvxMatrix> mats = tensor.Matrices;
            return new FlattenedMatricesAsVector(mats);
        }
    }

}
