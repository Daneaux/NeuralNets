using MatrixLibrary;
using NeuralNets;
using System.Diagnostics;

namespace MnistReader_ANN
{
    public class MNISTTrainingSet : ITrainingSet
    {
        public MNISTTrainingSet()
        {
            this.NumClasses = 10;

            int width = 0;
            int height = 0;
            int numLabels = 0;
            int numSamples = 0;

            MnistReader.GetMNISTTrainingMetaData(out numSamples, out numLabels, out width, out height);

            Width = width;
            Height = height;
            Depth = 1;
            NumberOfLabels = numLabels;
            NumberOfSamples = numSamples;
            OutputShape = new InputOutputShape(Width, Height, Depth, 1);
        }

        public int Width { get; }
        public int Height { get; }
        public int Depth { get; }
        public int NumClasses { get; private set; }
        public int NumberOfSamples { get; private set; }
        public int NumberOfLabels { get; private set; }
        public InputOutputShape OutputShape { get; }

        public List<TrainingPair> TrainingList { get; private set; }
        private List<Image> ImageList { get; set; }

        public List<TrainingPair> BuildNewRandomizedTrainingList(bool do2DImage = false)
        {
            Random rnd = new Random();
            List<TrainingPair> trainingPairs;
            if (this.ImageList == null)
            {
                this.ImageList = MnistReader.ReadTrainingData(do2DImage).ToList();
                Debug.Assert(ImageList.Count == 60000);
            }
            trainingPairs = new List<TrainingPair>((int)ImageList.Count);
            foreach (Image image in ImageList)
            {
                trainingPairs.Add(TrainingPairFromImage(image));
            }

            this.TrainingList = trainingPairs.OrderBy(x => rnd.Next()).ToList();
            return this.TrainingList;
        }

        private TrainingPair TrainingPairFromImage(Image image)
        {
            Tensor imageTensor;
            if (image is Normalized1DImage)
            {
                imageTensor = new AvxColumnVector((image as Normalized1DImage).Data).ToTensor();
            }
            else
            {
                imageTensor = new AvxMatrix((image as Normalized2DImage).Data).ToTensor();
            }
            AvxColumnVector outputVector = OneHotEncodeLabelData(image);
            TrainingPair trainingPair = new TrainingPair(imageTensor, outputVector.ToTensor());
            return trainingPair;
        }

        private AvxColumnVector OneHotEncodeLabelData(Image image)
        {
            // convert the label data (0,1,2, ...) into a columnvector. if the label is 7 (ie: byte == 7), then set the 7th float to 1.0
            float[] labelData = new float[this.NumClasses];
            labelData[(int)image.Label] = 1;
            return new AvxColumnVector(labelData);
        }

        private static AvxColumnVector ImageDataToColumnVector(Image image)
        {
            return null; // new AvxColumnVector(image.Data);
            /*
            // convert the image data into a columnvector
            float[] imageData = new float[image.Size];
            int i = 0;
            foreach (byte b in image.Data)
            {
                imageData[i++] = (float)b;
            }

            return new AvxColumnVector(imageData);*/
        }
    }
}
