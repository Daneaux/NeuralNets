using NeuralNets;

namespace NumReaderNetwork
{
    public class MNISTSpecificANN : GeneralFeedForwardANN
    {
        public MNISTSpecificANN(int hiddenLayerDim, int outputDim, double trainingRate) : base(28*28, trainingRate)
        {
            this.OutputLayerDim = outputDim;
            this.LossFunction = new CategoricalCrossEntropy();
            WeightedLayers.Add(new WeightedLayer(128, new ReLUActivaction(), InputDim));
            WeightedLayers.Add(new WeightedLayer(64, new SigmoidActivation(), 128));
            WeightedLayers.Add(new WeightedLayer(32, new SigmoidActivation(), 64));
            WeightedLayers.Add(new WeightedLayer(OutputLayerDim, new SoftMax(), 32));
        }
        public void TrainWithImages(int epochs)
        {
            for (int i = 0; i < epochs; i++)
            {
                int j = 0;
                foreach (Image image in MnistReader.ReadTrainingData())
                {
                    ColumnVector inputVector = ImageDataToColumnVector(image);
                    ColumnVector outputVector = OneHotEncodeLabelData(image);
                    TrainingPair trainingPair = new TrainingPair(inputVector, outputVector);

                    ColumnVector prediction = this.FeedForward(inputVector);
                    BackProp(trainingPair, prediction);

                    double error = GetAveragelLoss(trainingPair, prediction);
                    if (j % 100 == 0)
                    {
                        Console.WriteLine($"Epoch {i}, image {j}: Loss = {error}\n");
                    }
                    j++;
                }
            }
        }

        private ColumnVector OneHotEncodeLabelData(Image image)
        {
            // convert the label data (0,1,2, ...) into a columnvector. if the label is 7 (ie: byte == 7), then set the 7th double to 1.0
            double[] labelData = new double[this.OutputLayerDim];
            labelData[(int)image.Label] = 1.0;
            return new ColumnVector(labelData);
        }

        private static ColumnVector ImageDataToColumnVector(Image image)
        {
            // convert the image data into a columnvector
            double[] imageData = new double[image.Size];
            int i = 0;
            foreach (byte b in image.Data)
            {
                imageData[i++] = (double)b;
            }

            return new ColumnVector(imageData);
        }

    }
}
