﻿using System.Diagnostics;
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
    }

    public class AvxMatrix
    {
        private readonly float[,] Mat;
        public float this[int r, int c]
        {
            get { return this.Mat[r, c]; }
            set { this.Mat[r, c] = value; }
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

        public static AvxMatrix operator+(AvxMatrix lhs, AvxMatrix rhs) => lhs.AddMatrix(rhs);

        // [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public unsafe AvxMatrix AddMatrix(AvxMatrix b)
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
                        mat1 = m1 + r * this.Cols;
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
        public unsafe AvxMatrix MatrixTimesMatrix_TransposedRHS(AvxMatrix rhs)
        {
            // 
            // assumes transposed rhs
            // 
            Debug.Assert(this.Cols == rhs.Cols);
            const int floatsPerVector = 16;
            int numVectorsPerColumnRHS = rhs.Rows / floatsPerVector;
            int remainingVectorsPerColumnRHS = rhs.Rows % floatsPerVector;

            int numVecPerRowLHS = this.Cols / floatsPerVector;
            int remainingVecPerRowLHS = this.Cols % floatsPerVector;

            // result is still in the correct shape, but rhs is transposed.
            AvxMatrix result = new AvxMatrix(this.Rows, rhs.Cols);

            fixed (float* m1 = this.Mat,
                          m2 = rhs.Mat,
                          d1 = result.Mat)
            {
                float* mat1 = m1;
                float* mat2 = m2;
                float* dest = d1;

                for (int r = 0; r < Rows; r++)
                {
                    for (int rhsRowIndex = 0; rhsRowIndex < this.Rows; rhsRowIndex++)
                    {
                        // note: we're re-reading the same row many times. is there a way to cache it? next version: use r1 partial against all the corresponding c1's
                        mat1 = m1 + r * this.Cols; // point to the beginning of the same lhs row (until we increment r)

                        mat2 = m2 + rhsRowIndex * this.Cols; // point to the biggingin of the next rhs row

                        float v1DotCol = 0;
                        for (int c = 0; c < numVecPerRowLHS; c++, mat1 += 16, mat2 += 16)
                        {
                            Vector512<float> lhsRow = Vector512.Load<float>(mat1);
                            Vector512<float> rhsRow = Vector512.Load<float>(mat2);
                            v1DotCol += Vector512.Dot(lhsRow, rhsRow);
                        }

                        // do remainder
                        for (int i = 0; i < remainingVecPerRowLHS; i++, mat1++, mat2++)
                        {
                            v1DotCol += (*mat1) * (*mat2);
                        }
                        *dest = v1DotCol;
                        dest++;
                    }
                }
            }
            return result;
        }

        public AvxMatrix MatrixMultiply_Tiled(AvxMatrix rhs)
        {
            // how many tiles? how much left over on the edges?
            // i'm so done with columns and rows. we're doign to use W and H! that's it dammit, wdith (columns) and height (rows).
            const int tileSize = 16;

            int w1 = this.Cols;
            int h1 = this.Rows;

            int w2 = rhs.Cols;
            int h2 = rhs.Rows;

            int tw1 = w1 / tileSize;
            int th1 = h1 / tileSize;
            int tw2 = w2 / tileSize;
            int th2 = h2 / tileSize;

            int remainderW1 = w1 % tileSize;
            int remainderH2 = h2 % tileSize;

            Debug.Assert(tw1 == th2);

            AvxMatrix result = new AvxMatrix(this.Rows, rhs.Cols);

            // for each tile width
            for (int lhsRow = 0; lhsRow < th1; lhsRow++)
            {
                for (int rhsCol = 0; rhsCol < tw1; rhsCol++)
                {
                    // multiply row Y times column X (but using tiles)
                    DoRowColMultipleTileProduct(result, this, tileSize, rhs, rhsCol, lhsRow, tw1);
                }
            }

            // TODO: BUG: remainder!
            AvxMatrix.SubMatrixMultiplyNaive(
                this, rhs,
                0, 0,
                0, 0,
                0, 0,
                0, 0,
                result);               


            return result;
        }

        private static void SubMatrixMultiplyNaive(
            AvxMatrix lhs, AvxMatrix rhs, 
            int lhsX, int lhsY, 
            int rhsX, int rhsY, 
            int subMatrixWidthLHS, int subMatrixHeightLHS,
            int subMatrixWidthRHS, int subMatrixHeightRHS,  // terrible parameter names
            AvxMatrix dest)
        {
            for(int lhsRow = lhsY; lhsRow < lhsY + subMatrixHeightLHS; lhsRow++)
            {
                for(int rhsCol = rhsX; rhsCol < (rhsX + subMatrixWidthRHS); rhsCol++)
                {
                    Debug.Assert(subMatrixWidthLHS == subMatrixHeightRHS);
                    for(int i = 0; i < subMatrixHeightLHS; i++)
                    {
                        // lhs index
                        int lx = lhsRow;
                        int ly = lhsX + i;

                        // rhs index
                        int rx = rhsX;
                        int ry = rhsY + i;

                        dest[lhsRow, rhsCol] += lhs[ly, lx] * rhs[ry, rx];       // i think?                   
                    }
                }
            }
        }

        // naive implementation for remaining rows/columns
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private float DoRowTimesColumn(int myRow, int rightMatrixCol, AvxMatrix rightMatrix)
        {
            float cum = 0;
            for (int i = 0; i < Cols; i++)
            {
                cum += this[myRow, i] * rightMatrix[i, rightMatrixCol];
            }
            return cum;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private void DoRowColMultipleTileProduct(AvxMatrix dest, AvxMatrix lhs, int tileSize, AvxMatrix rhs, int x, int y, int numTiles)
        {
            // tile row y, and tile column y: multiply all tiles (across the row and down the rhs column, as though they're floats).
            // is the tile dot product!

            for (int tileNumber = 0; tileNumber < numTiles; tileNumber++)
            {
                // 1 - Grab the tile as vectors from the LHS matrix
                Vector512<float>[] lhsTile = GetHorizontalTileVectorArray(this, tileSize, y, tileNumber);
                // 2 - Grab the tile as vectors from the RHS matrix
                Vector512<float>[] rhsTile = GetVerticalTileVectorArray(rhs, tileSize, x, tileNumber);
                // 3 - dot product and accumulate
                DoOneTileDotProductAndStore(dest, lhsTile, rhsTile, tileSize, x, y);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private void DoOneTileDotProductAndStore(AvxMatrix dest, Vector512<float>[] lhsTile, Vector512<float>[] rhsTile, int tileSize, int x, int y)
        {
            int XStride = dest.Cols;
            // figure out where dest goes (we have the x,y tile coordinate, which we need to map to the w,h FLOAT starting point in the destination matrix
            // dot product all the vectors to generate another 16x16 dest matrix, a tile, and store it.
            Debug.Assert(lhsTile.Length == rhsTile.Length);
            Debug.Assert(lhsTile.Length == tileSize);

            float[,] tempResult = new float[tileSize, tileSize];
            for(int yy = 0; yy < tileSize; yy++)
            {
                for(int xx = 0; xx < tileSize; xx++)
                {
                    dest[yy, xx] += Vector512.Dot(lhsTile[yy], rhsTile[xx]);
                }
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe Vector512<float>[] GetHorizontalTileVectorArray(AvxMatrix lhs, int tileSize, int tileRowIndex, int tileNumber)
        {
            // x and y are the "tile" col and row, but not an actual index into the matrix.
            // so if y is 3, and the tileSize is 16, then we skip 3x16 rows, and each row is Col floats, so we skip: col*3*16 floats. to get the top left of the tile.
            // then, tileNumber tells us how many tiles to move into the row. 
            // so we add tileNumer * tileSize to our pointer.

            Vector512<float>[] vectors = new Vector512<float>[tileSize];

            fixed (float* m1 = lhs.Mat)
            {
                float* topLeftOfTile = m1 + (tileRowIndex * tileSize * lhs.Cols);
                topLeftOfTile += tileSize * tileNumber;

                // now get 16 float Vector512's. Skip col stride for each vector
                float* topLeft = topLeftOfTile;
                for (int i = 0; i < tileSize; i++, topLeft += lhs.Cols)
                {
                    vectors[i] = Vector512.Load<float>(topLeft);
                }
            }
            return vectors;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe Vector512<float>[] GetVerticalTileVectorArray(AvxMatrix rhs, int tileSize, int tileColIndex, int tileNumber)
        {
            // x and y are the "tile" col and row, but not an actual index into the matrix.
            // so if y is 3, and the tileSize is 16, then we skip 3x16 rows, and each row is Col floats, so we skip: col*3*16 floats. to get the top left of the tile.
            // then, tileNumber tells us how many tiles to move down into the column. 
            // so we add tileNumer * Cols * tileSize to our pointer.

            Vector512<float>[] vectors = new Vector512<float>[tileSize];

            fixed (float* m1 = rhs.Mat)
            {
                // each float array represents a verical slice on the tile. ie: we're tranposing in place
                float[][] vecs = new float[tileSize][];
                for (int i = 0; i < tileSize; i++)
                {
                    vecs[i] = new float[tileSize];
                }

                float* topLeftOfTile = m1 + (tileNumber * tileSize * rhs.Cols) + (tileColIndex * tileSize);                

                // now get 16 float Vector512's. Skip col stride for each vector. collect vertical vectors
                float* topLeft = topLeftOfTile;
                for (int i = 0; i < tileSize; i++, topLeft += rhs.Cols)
                {    
                    // j = vertical slice 
                    // i = horizontal index. (ie: which row)
                    for (int j = 0; j < tileSize; j++)
                    {
                        vecs[j][i] = *(topLeft+j);
                    }
                }

                // copy to Vector512.
                for (int i = 0; i < tileSize; i++)
                {
                    vectors[i] = Vector512.Create<float>(vecs[i]);
                }
            }
            return vectors;
        }


        // TODO: finish
        private static unsafe AvxMatrix Transpose(AvxMatrix matrix)
        {
            AvxMatrix result = new AvxMatrix(matrix.Cols, matrix.Rows);
            fixed (float* m1 = matrix.Mat)
            {
                
            }
            return result;
        }

    }
}