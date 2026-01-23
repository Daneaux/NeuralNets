using MatrixLibrary.BaseClasses;

namespace MatrixLibrary
{
    /// <summary>
    /// GPU-accelerated row vector implementation (stub - throws NotImplementedException).
    /// This stub allows the factory to create GPU instances without requiring
    /// a full GPU implementation. In Phase 3, these methods will be
    /// replaced with actual CUDA/OpenCL implementations.
    /// </summary>
    public class GpuRowVector : RowVectorBase
    {
        private readonly float[] _data;

        public int Size { get; }
        public float[] Row => _data;
        public MatrixBackend Backend => MatrixBackend.GPU;

        public GpuRowVector(int size)
        {
            Size = size;
            _data = new float[size];
        }

        public GpuRowVector(float[] data)
        {
            _data = data;
            Size = data.Length;
        }

        public float this[int i]
        {
            get => throw new NotImplementedException("GPU backend not yet implemented - use AVX or Software backend");
            set => throw new NotImplementedException("GPU backend not yet implemented - use AVX or Software backend");
        }

        public float Sum()
        {
            throw new NotImplementedException("GPU backend not yet implemented - use AVX or Software backend");
        }
    }
}
