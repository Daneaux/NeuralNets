using MatrixLibrary.BaseClasses;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace MatrixLibrary
{
    public class AvxColumnVector : ColumnVectorBase
    {
		public AvxColumnVector(float[] column) : base(column) { }
        public AvxColumnVector(int size) : base(size) { }
        protected AvxColumnVector() { }
        public override unsafe float Sum()
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
		public override AvxColumnVector Log()
		{
            return new ColumnVector(this.column).Log().ToAvxVector();
        }

		public override unsafe AvxColumnVector Multiply(float scalar)
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

		public override unsafe AvxColumnVector Add(float scalar)
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

		public override unsafe AvxColumnVector Subtract(float scalar)
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

		public override unsafe AvxColumnVector Multiply(ColumnVectorBase rhs)
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

		public override unsafe AvxColumnVector Add(ColumnVectorBase rhs)
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

		public override unsafe AvxColumnVector Subtract(ColumnVectorBase rhs)
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

		public override AvxMatrix RhsOuterProduct(Tensor lhs)
		{
			AvxColumnVector lhsVec = lhs.ToColumnVector() as AvxColumnVector;
			return lhsVec.OuterProduct(this);
		}

		// Effectively multiply column times vector and get outer product. doesn't matter if it's really a column or a row though.
		// the rhs is just a vector, and the outerproduct calc is the same regardless of what we call that vector.
		public override unsafe AvxMatrix OuterProduct(ColumnVectorBase rhs)
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

		public override AvxMatrix OuterProduct(FlattenedMatricesAsVector rhs)
		{
            throw new NotImplementedException();
			//var column = rhs.UnrollMatricesToColumnVector();
            //return OuterProduct(column);
		}
	}
}
