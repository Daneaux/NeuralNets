using MatrixLibrary.BaseClasses;

namespace MatrixLibrary
{
    /// <summary>
    /// GPU-accelerated column vector implementation (stub - throws NotImplementedException).
    /// This stub allows the factory to create GPU instances without requiring
    /// a full GPU implementation. In Phase 3, these methods will be
    /// replaced with actual CUDA/OpenCL implementations.
    /// </summary>
    public class GpuColumnVector : ColumnVectorBase
    {
        private readonly float[] _data;

        public int Size { get; }
        public float[] Column => _data;
        public MatrixBackend Backend => MatrixBackend.GPU;

        public GpuColumnVector(int size)
        {
            Size = size;
            _data = new float[size];
        }

        public GpuColumnVector(float[] data)
        {
            _data = data;
            Size = data.Length;
        }

        public override float Sum()
        {
            throw new NotImplementedException();
        }

        public override ColumnVectorBase Log()
        {
            throw new NotImplementedException();
        }

        public override ColumnVectorBase Multiply(float scalar)
        {
            throw new NotImplementedException();
        }

        public override ColumnVectorBase Multiply(ColumnVectorBase rhs)
        {
            throw new NotImplementedException();
        }

        public override ColumnVectorBase Add(float scalar)
        {
            throw new NotImplementedException();
        }

        public override ColumnVectorBase Add(ColumnVectorBase rhs)
        {
            throw new NotImplementedException();
        }

        public override ColumnVectorBase Subtract(float scalar)
        {
            throw new NotImplementedException();
        }

        public override ColumnVectorBase Subtract(ColumnVectorBase rhs)
        {
            throw new NotImplementedException();
        }

        public override MatrixBase RhsOuterProduct(Tensor lhs)
        {
            throw new NotImplementedException();
        }

        public override MatrixBase OuterProduct(ColumnVectorBase rhs)
        {
            throw new NotImplementedException();
        }

        public override MatrixBase OuterProduct(FlattenedMatricesAsVector rhs)
        {
            throw new NotImplementedException();
        }
    }
}
