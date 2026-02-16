using MatrixLibrary;
using MatrixLibrary.BaseClasses;
using System.Diagnostics;

namespace NeuralNets
{
    /// <summary>
    /// RenderContext specifically designed for Convolutional Neural Networks (CNNs).
    /// Unlike RenderContext which flattens 2D images, this class preserves spatial dimensions
    /// for proper CNN operations (convolution, pooling on 2D matrices).
    /// </summary>
    public class ConvolutionRenderContext
    {
        // Core properties
        public GeneralFeedForwardANN Network { get; }
        public int BatchSize { get; }
        public int CurrentThreadID { get; private set; }
        public ITrainingSet TrainingSet { get; }

        // Network accessors
        public int InputDim => Network.InputDim;
        public int OutputDim => Network.OutputDim;
        public int LayerCount => Network.LayerCount;
        public float LearningRate => Network.LearningRate;
        public Layer OutputLayer => Network.OutputLayer;
        public ILossFunction LossFunction => Network.LossFunction;
        public List<Layer> Layers => Network.Layers;

        // Context storage - uses Tensor to support 2D matrices (for CNNs) or 1D vectors (for dense layers)
        public Tensor[] ActivationContext { get; }
        public Tensor[] DerivativeContext { get; }

        // Logging interval - log progress every N batches
        private const int LogInterval = 500;

        /// <summary>
        /// Creates a new ConvolutionRenderContext for training CNNs with mini-batch gradient descent.
        /// </summary>
        /// <param name="network">The neural network to train</param>
        /// <param name="batchSize">Number of samples per batch</param>
        /// <param name="trainingSet">The training dataset (must provide 2D images)</param>
        public ConvolutionRenderContext(GeneralFeedForwardANN network, int batchSize, ITrainingSet trainingSet)
        {
            Network = network;
            this.BatchSize = batchSize;
            this.TrainingSet = trainingSet;
            this.ActivationContext = new Tensor[this.LayerCount];
            this.DerivativeContext = new Tensor[this.LayerCount];
            this.CurrentThreadID = Thread.CurrentThread.ManagedThreadId;
        }

        /// <summary>
        /// Creates a new ConvolutionRenderContext with default batch size of 64.
        /// </summary>
        /// <param name="network">The neural network to train</param>
        /// <param name="trainingSet">The training dataset (must provide 2D images)</param>
        public ConvolutionRenderContext(GeneralFeedForwardANN network, ITrainingSet trainingSet) : this(network, 64, trainingSet)
        {
        }

        /// <summary>
        /// Trains the network for multiple epochs using mini-batch gradient descent.
        /// </summary>
        /// <param name="numEpochs">Number of epochs to train</param>
        public void EpochTrain(int numEpochs)
        {
            for (int i = 0; i < numEpochs; i++)
            {
                BatchTrain(this, i);
            }
        }

        /// <summary>
        /// Performs mini-batch gradient descent training for CNNs.
        /// For each batch:
        /// 1. Resets gradient accumulators on all layers
        /// 2. Processes batchSize samples with 2D images, accumulating gradients
        /// 3. Averages gradients and updates weights once per batch
        /// 
        /// Note: Uses single-threaded execution for thread safety with CNN layers that store per-sample state.
        /// </summary>
        /// <param name="parentContext">The context containing network and training parameters</param>
        /// <param name="epochNum">Current epoch number (for logging)</param>
        public static void BatchTrain(ConvolutionRenderContext parentContext, int epochNum)
        {
            // CRITICAL: Use do2DImage=true to preserve 2D spatial dimensions for CNN operations
            bool do2dImage = true;
            List<TrainingPair> trainingPairs = parentContext.TrainingSet.BuildNewRandomizedTrainingList(do2dImage);
            int totalSamples = parentContext.TrainingSet.NumberOfSamples;
            int maxBatches = totalSamples / parentContext.BatchSize;

            int currentSampleIndex = 0;

            for (int batchIdx = 0; batchIdx < maxBatches; batchIdx++)
            {
                // Reset accumulators at start of each batch
                foreach (Layer layer in parentContext.Network.Layers)
                {
                    layer.ResetAccumulators();
                }

                // Process batchSize samples
                // Each sample: forward pass + backward pass to accumulate gradients
                int batchStartIndex = currentSampleIndex;

                // Single-threaded execution (safer for CNN layers that store per-sample state)
                for (int sampleIdx = 0; sampleIdx < parentContext.BatchSize; sampleIdx++)
                {
                    // Get the training pair for this sample
                    TrainingPair trainingPair;
                    int sampleIndex = batchStartIndex + sampleIdx;
                    trainingPair = trainingPairs[sampleIndex];

                    // Forward pass - preserves 2D tensors through CNN layers
                    Tensor predictedOut = FeedForwardStatic(parentContext.Network.Layers, trainingPair.Input);

                    // Backward pass - accumulates gradients into shared layer accumulators
                    BackPropStatic(parentContext.Network, trainingPair, predictedOut);
                }

                // Update weights once per batch using accumulated (and averaged) gradients
                foreach (Layer layer in parentContext.Network.Layers)
                {
                    layer.UpdateWeightsAndBiasesWithScaledGradients(parentContext.LearningRate);
                }

                // Log progress every N batches
                if (batchIdx % LogInterval == 0)
                {
                    // Use the last sample of this batch for loss calculation
                    TrainingPair sampleForLoss = trainingPairs[currentSampleIndex + parentContext.BatchSize - 1];
                    Tensor predictedOut = FeedForwardStatic(parentContext.Network.Layers, sampleForLoss.Input);
                    float totalLoss = parentContext.Network.GetTotallLoss(sampleForLoss, predictedOut.ToColumnVector());
                    Console.WriteLine($"Epoch {epochNum}, batch size:{parentContext.BatchSize}. Finished Batch {batchIdx}/{maxBatches} with total loss = {totalLoss:F6}");
                }

                currentSampleIndex += parentContext.BatchSize;
            }
        }

        /// <summary>
        /// Static version of FeedForward that doesn't require creating a ConvolutionRenderContext.
        /// Handles 2D tensors (for CNNs) and 1D vectors (for dense layers).
        /// </summary>
        /// <param name="layers">List of layers to process</param>
        /// <param name="input">Input tensor (2D for CNN start, 1D for dense network)</param>
        /// <returns>Output tensor from final layer</returns>
        private static Tensor FeedForwardStatic(List<Layer> layers, Tensor input)
        {
            Tensor lastOutput = input;
            foreach (Layer layer in layers)
            {
                lastOutput = layer.FeedFoward(lastOutput);
            }
            return lastOutput;
        }

        /// <summary>
        /// Static version of BackProp that doesn't require creating a ConvolutionRenderContext.
        /// Accumulates gradients directly into the shared network layers.
        /// Handles both 2D (CNN) and 1D (dense) tensors correctly.
        /// </summary>
        /// <param name="network">The neural network</param>
        /// <param name="trainingPair">Training sample with input and expected output</param>
        /// <param name="predictedOut">Predicted output from forward pass</param>
        private static void BackPropStatic(NeuralNetworkAbstract network, TrainingPair trainingPair, Tensor predictedOut)
        {
            // Compute loss derivative
            Tensor dE_dX = network.LossFunction.Derivative(trainingPair.Output.ToColumnVector(), predictedOut.ToColumnVector()).ToTensor();

            // Backpropagate through all layers in reverse order
            foreach (Layer layer in network.Layers.Reverse<Layer>())
            {
                // All layers (including activation and convolution) handle their own derivative computation
                dE_dX = layer.BackPropagation(dE_dX);
            }
        }

        /// <summary>
        /// Performs forward pass through all layers starting from the given input.
        /// Stores intermediate activations in ActivationContext.
        /// </summary>
        /// <param name="input">Input tensor (2D for CNN images, 1D for dense vectors)</param>
        /// <returns>Output tensor from the final layer</returns>
        public Tensor FeedForward(Tensor input)
        {
            Tensor lastActivation = input;
            for (int i = 0; i < this.LayerCount; i++)
            {
                Layer currentLayer = Layers[i];
                Tensor output = currentLayer.FeedFoward(lastActivation);
                lastActivation = output;
                this.SetLastActivation(i, lastActivation);
            }
            return lastActivation;
        }

        /// <summary>
        /// Performs backward pass (backpropagation) given a training pair and predicted output.
        /// Computes gradients and accumulates them into layer accumulators.
        /// </summary>
        /// <param name="trainingPair">Training sample with input and expected output</param>
        /// <param name="predicted">Predicted output from forward pass</param>
        public void BackProp(TrainingPair trainingPair, Tensor predicted)
        {
            // Compute loss derivative
            Tensor dE_dX = LossFunction.Derivative(trainingPair.Output.ToColumnVector(), predicted.ToColumnVector()).ToTensor();

            // Backpropagate through all layers in reverse order
            int layerIndex = this.Layers.Count - 1;
            foreach (Layer layer in this.Layers.Reverse<Layer>())
            {
                dE_dX = layer.BackPropagation(dE_dX);
                this.SetLastDerivative(layerIndex, dE_dX);
                layerIndex--;
            }
        }

        /// <summary>
        /// Calculates total loss for a training pair given the predicted output.
        /// </summary>
        /// <param name="pair">Training pair with expected output</param>
        /// <param name="predicted">Predicted output tensor</param>
        /// <returns>Total loss value</returns>
        public float GetTotallLoss(TrainingPair pair, Tensor predicted)
        {
            return Network.GetTotallLoss(pair, predicted.ToColumnVector());
        }

        /// <summary>
        /// Calculates total loss for a training pair given the predicted output as ColumnVector.
        /// </summary>
        /// <param name="pair">Training pair with expected output</param>
        /// <param name="predicted">Predicted output column vector</param>
        /// <returns>Total loss value</returns>
        public float GetTotallLoss(TrainingPair pair, ColumnVectorBase predicted)
        {
            return Network.GetTotallLoss(pair, predicted);
        }

        // Private helper methods for context management
        private void SetLastActivation(int layerIndex, Tensor lastActivation)
        {
            Debug.Assert(layerIndex >= 0);
            ActivationContext[layerIndex] = lastActivation;
        }

        private void SetLastDerivative(int layerIndex, Tensor derivative)
        {
            Debug.Assert(layerIndex >= 0);
            DerivativeContext[layerIndex] = derivative;
        }
    }
}
