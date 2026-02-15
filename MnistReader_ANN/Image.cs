
namespace MnistReader_ANN
{
    public abstract class Image
    {
        public virtual byte Label { get; set; }
        public abstract int Size { get; }
    }
}
