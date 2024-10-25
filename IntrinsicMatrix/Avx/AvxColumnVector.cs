using System.Runtime.Intrinsics.X86;
using System.Runtime.Intrinsics;

namespace MatrixLibrary
{
    public class AvxColumnVector
    {
        private readonly float[] column;
        public virtual int Size { get { return column.Length; } }

        public float this[int a] => column[a];
        public float[] Column { get { return column; } }

        public AvxColumnVector(float[] column)
        {
            this.column = column;
        }
        public AvxColumnVector(int size)
        {
            this.column = new float[size];
        }

        public float GetMax()
        {
            float max = float.MinValue;
            for(int i = 0; i < column.Length; i++)
            {
                if(column[i] > max)
                    max = column[i];
            }
            return max;
        }

        protected AvxColumnVector() { }
        public unsafe float Sum()
        {
            float sum = 0.0f;
            const int floatsPerVector = 16;
            int size = this.Size;
            int numVectors = size / floatsPerVector;
            int remainingElements = size % floatsPerVector;

            fixed (float* c1 = this.column)
            {
                float* col = c1;

                for (int i = 0; i < numVectors; i++, col += 16)
                {
                    Vector512<float> v1 = Vector512.Load<float>(col);
                    sum += Vector512.Sum<float>(v1);
                }

                // do remainder
                for (int i = 0; i < remainingElements; i++, col++)
                {
                    sum += *col;
                }
            }

            return sum;
        }


        // todo: can't find an AVX512 (or any other intrinsic) to do simd logarithm
        public AvxColumnVector Log()
        {
            return new ColumnVector(this.column).Log().ToAvxVector();
        }

        public static AvxColumnVector operator *(AvxMatrix lhs, AvxColumnVector rhs) => rhs.MatrixTimesColumn(lhs);

        public unsafe virtual AvxColumnVector MatrixTimesColumn(AvxMatrix lhs)
        {
            const int floatsPerVector = 16;
            int numVectorsPerRow = lhs.Cols / floatsPerVector;
            int remainingColumns = lhs.Cols % floatsPerVector;

            AvxColumnVector result = new AvxColumnVector(lhs.Rows);

            fixed (float* m1 = lhs.Mat,
                          col = this.Column,
                          resCol = result.Column)
            {
                float* mat1 = m1;
                float* col1 = col;
                float* destCol = resCol;

                for (int r = 0; r < lhs.Rows; r++, destCol++)
                {
                    col1 = col;             // restart at the top of the column vector
                    float v1DotCol = 0;
                    for (int c = 0; c < numVectorsPerRow; c++, mat1 += 16, col1 += 16)
                    {
                        Vector512<float> rhsVec = Vector512.Load<float>(col1);
                        Vector512<float> lhsVec = Vector512.Load<float>(mat1);
                        v1DotCol += Vector512.Dot(lhsVec, rhsVec);
                    }

                    // do remainder
                    for (int i = 0; i < remainingColumns; i++, mat1++, col1++)
                    {
                        v1DotCol += (*mat1) * (*col1);
                    }
                    *destCol = v1DotCol;
                }
            }
            return result;
        }

        public static AvxColumnVector operator *(AvxColumnVector vec, float scalar) => vec.ScalarMultiply(scalar);
        public static AvxColumnVector operator *(float scalar, AvxColumnVector vec) => vec.ScalarMultiply(scalar);

        public unsafe AvxColumnVector ScalarMultiply(float scalar)
        {
            AvxColumnVector lhs = this;
            AvxColumnVector result = new AvxColumnVector(lhs.Size);
            const int vector512Floats = 16;

            int vecSize = lhs.Size / vector512Floats;
            int vecRemainder = lhs.Size % vector512Floats;

            fixed (float* av = lhs.Column,
                          rv = result.Column)
            {
                float* col1 = av;
                float* destCol = rv;
                Vector512<float> v2 = Vector512.Create<float>(scalar);

                for (int i = 0; i < vecSize; i++, col1 += vector512Floats, destCol += vector512Floats)
                {
                    Vector512<float> v1 = Vector512.Load<float>(col1);
                    Vector512<float> v3 = Avx512F.Multiply(v1, v2);
                    Vector512.Store<float>(v3, destCol);
                }

                // remainder
                for (int i = 0; i < vecRemainder; i++, col1++, destCol++)
                {
                    *destCol = *col1 * scalar;
                }
            }

            return result;
        }

        public static AvxColumnVector operator +(AvxColumnVector vec, float scalar) => vec.ScalarAddition(scalar);
        public static AvxColumnVector operator +(float scalar, AvxColumnVector vec) => vec.ScalarAddition(scalar);
        public static AvxColumnVector operator -(AvxColumnVector vec, float scalar) => vec.ScalarAddition(-scalar);

        private unsafe AvxColumnVector ScalarAddition(float scalar)
        {
            AvxColumnVector lhs = this;
            AvxColumnVector result = new AvxColumnVector(lhs.Size);
            const int vector512Floats = 16;

            int vecSize = lhs.Size / vector512Floats;
            int vecRemainder = lhs.Size % vector512Floats;

            fixed (float* av = lhs.Column,
                          rv = result.Column)
            {
                float* col1 = av;
                float* destCol = rv;
                Vector512<float> v2 = Vector512.Create<float>(scalar);

                for (int i = 0; i < vecSize; i++, col1 += vector512Floats, destCol += vector512Floats)
                {
                    Vector512<float> v1 = Vector512.Load<float>(col1);
                    Vector512<float> v3 = Avx512F.Add(v1, v2);
                    Vector512.Store<float>(v3, destCol);
                }

                // remainder
                for (int i = 0; i < vecRemainder; i++, col1++, destCol++)
                {
                    *destCol = *col1 + scalar;
                }
            }

            return result;
        }

        public static AvxColumnVector operator -(float scalar, AvxColumnVector vec) => vec.ScalarSubtract(scalar);
        private unsafe AvxColumnVector ScalarSubtract(float scalar)
        {
            AvxColumnVector lhs = this;
            AvxColumnVector result = new AvxColumnVector(lhs.Size);
            const int vector512Floats = 16;

            int vecSize = lhs.Size / vector512Floats;
            int vecRemainder = lhs.Size % vector512Floats;

            fixed (float* av = lhs.Column,
                          rv = result.Column)
            {
                float* col1 = av;
                float* destCol = rv;
                Vector512<float> v2 = Vector512.Create<float>(scalar);

                for (int i = 0; i < vecSize; i++, col1 += vector512Floats, destCol += vector512Floats)
                {
                    Vector512<float> v1 = Vector512.Load<float>(col1);
                    Vector512<float> v3 = Avx512F.Subtract(v2, v1);
                    Vector512.Store<float>(v3, destCol);
                }

                // remainder
                for (int i = 0; i < vecRemainder; i++, col1++, destCol++)
                {
                    *destCol = scalar - *col1;
                }
            }

            return result;
        }


        public static AvxColumnVector operator *(AvxColumnVector lhs, AvxColumnVector rhs) => lhs.MultiplyVecVec(rhs);

        private unsafe AvxColumnVector MultiplyVecVec(AvxColumnVector rhs)
        {
            AvxColumnVector lhs = this;
            AvxColumnVector result = new AvxColumnVector(lhs.Size);
            const int vector512Floats = 16;

            int vecSize = lhs.Size / vector512Floats;
            int vecRemainder = lhs.Size % vector512Floats;

            fixed (float* av = lhs.Column,
                          bv = rhs.Column,
                          rv = result.Column)
            {
                float* col1 = av;
                float* col2 = bv;
                float* destCol = rv;

                for (int i = 0; i < vecSize; i++, col1 += vector512Floats, col2 += vector512Floats, destCol += vector512Floats)
                {
                    Vector512<float> v1 = Vector512.Load<float>(col1);
                    Vector512<float> v2 = Vector512.Load<float>(col1);
                    Vector512<float> v3 = Avx512F.Multiply(v1, v2);
                    Vector512.Store<float>(v3, destCol);
                }

                // remainder
                for (int i = 0; i < vecRemainder; i++, col1++, col2++, destCol++)
                {
                    *destCol = *col1 * *col2;
                }
            }

            return result;
        }

        public static AvxColumnVector operator +(AvxColumnVector lhs, AvxColumnVector rhs) => lhs.AddVecVec(rhs);

        private unsafe AvxColumnVector AddVecVec(AvxColumnVector rhs)
        {
            AvxColumnVector lhs = this;
            AvxColumnVector result = new AvxColumnVector(lhs.Size);
            const int vector512Floats = 16;

            int vecSize = lhs.Size / vector512Floats;
            int vecRemainder = lhs.Size % vector512Floats;

            fixed (float* av = lhs.Column,
                          bv = rhs.Column,
                          rv = result.Column)
            {
                float* col1 = av;
                float* col2 = bv;
                float* destCol = rv;

                for (int i = 0; i < vecSize; i++, col1 += vector512Floats, col2 += vector512Floats, destCol += vector512Floats)
                {
                    Vector512<float> v1 = Vector512.Load<float>(col1);
                    Vector512<float> v2 = Vector512.Load<float>(col1);
                    Vector512<float> v3 = Avx512F.Add(v1, v2);
                    Vector512.Store<float>(v3, destCol);
                }

                // remainder
                for (int i = 0; i < vecRemainder; i++, col1++, col2++, destCol++)
                {
                    *destCol = *col1 + *col2;
                }
            }

            return result;
        }

        public static AvxColumnVector operator -(AvxColumnVector lhs, AvxColumnVector rhs) => lhs.SubtractVecVec(rhs);

        private unsafe AvxColumnVector SubtractVecVec(AvxColumnVector rhs)
        {
            AvxColumnVector lhs = this;
            AvxColumnVector result = new AvxColumnVector(lhs.Size);
            const int vector512Floats = 16;

            int vecSize = lhs.Size / vector512Floats;
            int vecRemainder = lhs.Size % vector512Floats;

            fixed (float* av = lhs.Column,
                          bv = rhs.Column,
                          rv = result.Column)
            {
                float* col1 = av;
                float* col2 = bv;
                float* destCol = rv;

                for (int i = 0; i < vecSize; i++, col1 += vector512Floats, col2 += vector512Floats, destCol += vector512Floats)
                {
                    Vector512<float> v1 = Vector512.Load<float>(col1);
                    Vector512<float> v2 = Vector512.Load<float>(col1);
                    Vector512<float> v3 = Avx512F.Subtract(v1, v2);
                    Vector512.Store<float>(v3, destCol);
                }

                // remainder
                for (int i = 0; i < vecRemainder; i++, col1++, col2++, destCol++)
                {
                    *destCol = *col1 - *col2;
                }
            }

            return result;
        }

        public virtual AvxMatrix RhsOuterProduct(Tensor lhs)
        {
            AvxColumnVector lhsVec = lhs.ToAvxColumnVector();
            return lhsVec.OuterProduct(this);
        }

        // Effectively multiply column times vector and get outer product. doesn't matter if it's really a column or a row though.
        // the rhs is just a vector, and the outerproduct calc is the same regardless of what we call that vector.
        public unsafe AvxMatrix OuterProduct(AvxColumnVector rhs)
        {
            const int floatsPerVector = 16;
            int numVecPerRHS = rhs.Size / floatsPerVector;
            int remainingRHS = rhs.Size % floatsPerVector;

            AvxMatrix result = new AvxMatrix(this.Size, rhs.Size);

            fixed (float* c = this.Column,
                          r = rhs.Column,
                          rm = result.Mat)
            {
                float* col = c;
                float* row = r;
                float* resultMatrix = rm;

                for (int i = 0; i < this.Size; i++, col++)
                {
                    float lhsMUlt = *col;
                    row = r; // reset row pointer. Remember there's just one row and one column.
                    for (int j = 0; j < numVecPerRHS; j++, row += floatsPerVector, resultMatrix += floatsPerVector)
                    {
                        // loading the same row again and again ... maybe just keep them all?
                        Vector512<float> rhsVec = Vector512.Load<float>(row);
                        Vector512<float> mult = Vector512.Multiply<float>(lhsMUlt, rhsVec);
                        Vector512.Store<float>(mult, resultMatrix);
                    }

                    // do remainder of the row
                    for (int j = 0; j < remainingRHS; j++, row++, resultMatrix++)
                    {
                        *resultMatrix = lhsMUlt * (*row);
                    }
                }
            }

            return result;
        }

        public AvxMatrix OuterProduct(FlattenedMatricesAsVector rhs)
        {
            float[] floats = rhs.FlattenAllMatricesAndCopyUgh();
            return OuterProduct(new AvxColumnVector(floats));
        }

        public void SetRandom(int seed, int min, int max)
        {
            Random rnd = new Random(seed);
            float width = max - min;
            for (int r = 0; r < Size; r++)
            {
                column[r] = (float)((rnd.NextDouble() * width) + min);
            }
        }
    }
}
