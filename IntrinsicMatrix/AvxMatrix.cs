using System;
using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace IntrinsicMatrix
{
    public class AvxColumnVector
    {
        private readonly float[] column;
        public int Size { get { return column.Length; } }

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
    }

    public class AvxMatrix
    {
        private readonly float[,] Mat;
        public float this[int r, int c]
        {
            get { return this.Mat[r, c]; }
        }

        public AvxMatrix(float[,] mat) 
        {
            Mat = mat; 
            Rows = Mat.GetLength(0);
            Cols = Mat.GetLength(1);
        }

        public AvxMatrix(int rows, int cols)
        {
            Rows = rows;
            Cols = cols;
            Mat = new float[rows, cols];
        }
        public void SetRandom(int seed, float min, float max)
        {
            Random rnd = new Random(seed);
            float width = max - min;
            for (int c = 0; c < Cols; c++)
            {
                for (int r = 0; r < Rows; r++)
                {
                    Mat[r, c] = (float)((rnd.NextDouble() * width) + min);
                }
            }
        }

        public int Rows { get; private set; }
        public int Cols { get; private set; }

        public static AvxMatrix operator+(AvxMatrix lhs, AvxMatrix rhs) => lhs.AddAvx512(rhs);

        // [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public unsafe AvxMatrix AddAvx512(AvxMatrix b)
        {
            const int floatsPerVector = 16;
            int size = Rows * Cols;
            int numVectors = size / floatsPerVector;
            int remainingElements = size % floatsPerVector;

            AvxMatrix result = new AvxMatrix(Rows, Cols);

            fixed (float* m1 = this.Mat,
                          m2 = b.Mat,
                          d1 = result.Mat)
            {
                float* mat1 = m1;
                float* mat2 = m2;
                float* dest = d1;

                for (int i = 0; i < numVectors; i++, mat1 += 16, mat2 += 16, dest += 16)
                {
                    Vector512<float> v1 = Vector512.Load<float>(mat1);
                    Vector512<float> v2 = Vector512.Load<float>(mat2);
                    Vector512<float> sum = Avx512F.Add(v1, v2);
                    Vector512.Store<float>(sum, dest);
                }

                // do remainder
                for (int i = 0; i < remainingElements; i++, dest++, mat1++, mat2++)
                {
                    *dest = *mat1 + *mat2;
                }
            }

            return result;
        }

        public unsafe AvxColumnVector MatrixTimesColumn(AvxColumnVector b)
        {
            const int floatsPerVector = 16;
            int numVectorsPerRow = Cols / floatsPerVector;
            int remainingColumns = Cols % floatsPerVector;

            AvxColumnVector result = new AvxColumnVector(Rows);

            fixed (float* m1 = this.Mat,
                          col = b.Column,
                          resCol = result.Column)
            {
                float* mat1 = m1;
                float* col1 = col;
                float* destCol = resCol;

                Vector512<float> colVector = Vector512.Load<float>(col1);
                for (int r = 0; r < Rows; r++, destCol++)
                {
                    col1 = col;             // restart at the top of the column vector
                    float v1DotCol = 0;
                    for (int c = 0; c < numVectorsPerRow; c++, mat1 += 16, col1 += 16)
                    {
                        Vector512<float> v1 = Vector512.Load<float>(mat1);
                        v1DotCol += Vector512.Dot(v1, colVector);
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

        public static AvxMatrix operator *(AvxMatrix lhs, AvxMatrix rhs) => lhs.MatrixTimesMatrix(rhs);

        // no transpose, no tile
        private unsafe AvxMatrix MatrixTimesMatrix(AvxMatrix rhs)
        {
            Debug.Assert(this.Cols == rhs.Rows);
            const int floatsPerVector = 16;
            int numVectorsPerColumnRHS = rhs.Rows / floatsPerVector;
            int remainingVectorsPerColumnRHS = rhs.Rows % floatsPerVector;

            int numVecPerRowLHS = this.Cols / floatsPerVector;
            int remainingVecPerRowLHS = this.Cols % floatsPerVector;

            AvxMatrix result = new AvxMatrix(this.Rows, rhs.Cols);
            int mat2Stride = rhs.Cols;  // this skips a whole row, wraps around to the next element in the column.

            fixed (float* m1 = this.Mat,
                          m2 = rhs.Mat,
                          d1 = result.Mat)
            {
                float* mat1 = m1;
                float* mat2 = m2;
                float* dest = d1;

                // grab a lhs row.
                for (int r = 0; r < this.Rows; r++)
                {
                    //
                    // doing ONE row (left to right) on the LHS times ONE column on the RHS
                    //

                    // holding this ONE row steady, do every column
                    for (int c = 0; c < rhs.Cols; c++)
                    {
                        // reset Mat1 ... yes we're redoing the row in LHS every single time... it'd be better maybe to store it as a list of vectors? dunno.
                        mat1 = m1 +  r * this.Cols;
                        mat2 = m2 + c; // reset the column pointer to be at the top of the next column. which is 'start' plus number of columns we've done
                        float currentDot = 0;

                        // in this row, grab increments of 16 from the row and then do remainder
                        // stop when we're reach the end multiple of 16, then do remainder below
                        Debug.Assert(numVectorsPerColumnRHS == numVecPerRowLHS);
                        for (int vectorCount = 0; vectorCount < numVecPerRowLHS; vectorCount++, mat1 += floatsPerVector, mat2 += mat2Stride * floatsPerVector)
                        {
                            Vector512<float> lhsRow = Vector512.Load<float>(mat1);
                            Vector512<float> rhsCol = CollectColumnFromAt(mat2, mat2Stride, floatsPerVector);
                            currentDot += Vector512.Dot(lhsRow, rhsCol);
                        }

                        // do remainder
                        Debug.Assert(remainingVecPerRowLHS == remainingVectorsPerColumnRHS);
                        for (int i = 0; i < remainingVecPerRowLHS; i++, mat1++, mat2 += mat2Stride)
                        {
                            currentDot += (*mat1) * (*mat2);
                        }

                        // store in result[x,y]
                        *dest = currentDot;
                        dest++;
                    }                    
                }
            }

            return result;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private unsafe Vector512<float> CollectColumnFromAt(float *start, int stride, int howManytoTake)
        {
            Debug.Assert(howManytoTake >= 16);

            float[] floats = new float[16];
            for (int r = 0; r < howManytoTake; r++, start += stride)
            {
                floats[r] = *start;
            }

            return  Vector512.Create<float>(floats);
        }

        // try transpose, no tile
        private AvxMatrix MatrixTimesMatrix_TransposeRHS(AvxMatrix rhs)
        {
            Debug.Assert(this.Cols == rhs.Rows);
            AvxMatrix result = new AvxMatrix(rhs.Cols, this.Rows);

            return null;

        }

        // try tile
        private AvxMatrix MatrixTimesMatrix_Tiled(AvxMatrix rhs)
        {
            Debug.Assert(this.Cols == rhs.Rows);
            AvxMatrix result = new AvxMatrix(rhs.Cols, this.Rows);

            return null;
        }
    }
}
