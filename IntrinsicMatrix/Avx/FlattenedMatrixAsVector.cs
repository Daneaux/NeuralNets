using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Runtime.Intrinsics;
using System.Text;
using System.Threading.Tasks;

namespace MatrixLibrary
{
    public class FlattenedMatricesAsVector : AvxColumnVector
    {
        public List<AvxMatrix> AvxMatrices { get; }

        public override int Size => this.AvxMatrices.Count * AvxMatrices[0].TotalSize;

        // The crazy idea here is, you have a bunch of N x M matrices, but we 
        // need to unroll them and contatinate them into one long column vector
        // So... rather than copy each matrix into a new float[], which is super wasetful
        // we wrap all the matrices here and treat them like a long vector; for all the operations
        // If this was C++ I would just case the float[,] to a float[] and be done. But you can't in C#.
        public FlattenedMatricesAsVector(List<AvxMatrix> avxMatrices) : base()
        {
            AvxMatrices = avxMatrices;
        }
        public override AvxMatrix RhsOuterProduct(Tensor lhs)
        {
            AvxColumnVector rhs = new AvxColumnVector(FlattenAllMatricesAndCopyUgh());
            AvxColumnVector lhsVec = lhs.ToAvxColumnVector();
            return lhsVec.OuterProduct(rhs);
        }

        public static AvxColumnVector operator *(AvxMatrix lhs, FlattenedMatricesAsVector vec) => vec.MatrixTimesColumn(lhs);

        public unsafe override AvxColumnVector MatrixTimesColumn(AvxMatrix lhs)
        {
            AvxColumnVector result = new AvxColumnVector(lhs.Rows);

            Debug.Assert(this.AvxMatrices.Count * this.AvxMatrices[0].TotalSize == lhs.Cols);
            fixed (float* res_ = result.Column)
            {
                float* res = res_;
                for (int r = 0; r < lhs.Rows; r++, res++)
                {
                    this.MatrixTimesColumnPartial(lhs, r, this.AvxMatrices, res); 
                }
            }
            return result;
        }

        // So ugly, but for now, let's get this correct (namely the out product needs this)
        // and later optimize. Correctness first ... 
        public float[] FlattenAllMatricesAndCopyUgh()
        {
            float[] floats = new float[this.Size];
            int i = 0;
            foreach(AvxMatrix mat in AvxMatrices)
            {
                for (int r = 0; r < mat.Rows; r++)
                    for (int c = 0; c < mat.Cols; c++)
                        floats[i++] = mat[r, c];
            }
            return floats;
        }

        private unsafe void MatrixTimesColumnPartial(
            AvxMatrix lhs, 
            int lhsStartingRow, 
            List<AvxMatrix> flattenMatrices,
            float *destCol)
        {
            const int floatsPerVector = 16;
            int rhsColumnLength = flattenMatrices[0].TotalSize;
            int numVectorsPerRow = rhsColumnLength / floatsPerVector;
            int remainingColumns = rhsColumnLength % floatsPerVector;

            Debug.Assert(lhs.Cols >= rhsColumnLength);
            Debug.Assert(lhs.Cols % rhsColumnLength == 0);

            fixed (float* m1 = lhs.Mat)
            {
                float v1DotCol = 0;
                float* lhsMatPointer = m1 + (lhs.Cols * lhsStartingRow); // start of the row

                for (int rhsIndex = 0; rhsIndex < flattenMatrices.Count; rhsIndex++)
                {
                    Debug.Assert(flattenMatrices[rhsIndex].TotalSize == rhsColumnLength);

                    fixed (float* rhsCol_ = flattenMatrices[rhsIndex].Mat)
                    {
                        float* rhsCol = rhsCol_;

                        for (int c = 0; c < numVectorsPerRow; c++, lhsMatPointer += 16, rhsCol += 16)
                        {
                            Vector512<float> rhsVec = Vector512.Load<float>(rhsCol);
                            Vector512<float> lhsVec = Vector512.Load<float>(lhsMatPointer);
                            v1DotCol += Vector512.Dot(lhsVec, rhsVec);
                        }

                        // do remainder
                        for (int i = 0; i < remainingColumns; i++, lhsMatPointer++, rhsCol++)
                        {
                            v1DotCol += (*lhsMatPointer) * (*rhsCol);
                        }
                    }
                }
                *destCol = v1DotCol;
            }
        }
    }
}
