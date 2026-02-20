using MatrixLibrary;
using MatrixLibrary.BaseClasses;
using NeuralNets;
namespace NeuralNetsTests
{
    /// <summary>
    /// Diagnostic tests to identify CNN convergence issues.
    /// Each test isolates a specific aspect of the CNN pipeline to pinpoint
    /// where gradients or values go wrong compared to PyTorch.
    /// </summary>
    [TestClass]
    public class CNNConvergenceDiagnostics
    {
        #region Test 1: Conv bias initialization bug

        /// <summary>
        /// ConvolutionLayer.InitKernelsAndBiases calls bias.SetRandom(seed, -0.1f, -0.1f).
        /// When min == max, this sets ALL bias values to exactly -0.1.
        /// This is almost certainly a bug — the intent was likely (-0.1f, 0.1f).
        /// This test catches it by checking that biases are not all identical.
        /// </summary>
        [TestMethod]
        public void ConvLayer_BiasInitialization_ShouldNotBeConstant()
        {
            var inputShape = new InputOutputShape(8, 8, 1, 1);
            var conv = new ConvolutionLayer(inputShape, kernelCount: 2, kernelSquareDimension: 3, stride: 1);

            // Conv output for 8x8 input with 3x3 kernel, stride 1 = 6x6
            // Each bias is a 6x6 matrix (one per kernel)
            for (int k = 0; k < 2; k++)
            {
                MatrixBase bias = conv.Biases[k];
                float firstVal = bias[0, 0];
                bool allSame = true;

                for (int r = 0; r < bias.Rows; r++)
                    for (int c = 0; c < bias.Cols; c++)
                        if (System.Math.Abs(bias[r, c] - firstVal) > 1e-9f)
                            allSame = false;

                // If all values are identical, the initialization is wrong
                // SetRandom(-0.1f, -0.1f) produces all -0.1 values
                if (allSame)
                {
                    Console.WriteLine($"BUG DETECTED: Kernel {k} bias is constant ({firstVal}). " +
                        "SetRandom(seed, -0.1f, -0.1f) sets all values to -0.1. " +
                        "Should be SetRandom(seed, -0.1f, 0.1f).");
                }

                Assert.IsFalse(allSame,
                    $"Kernel {k}: All bias values are {firstVal}. " +
                    "This means SetRandom(min, max) was called with min==max. " +
                    "Fix: change bias.SetRandom(RandomSeed, -0.1f, -0.1f) to bias.SetRandom(RandomSeed, -0.1f, 0.1f) " +
                    "in ConvolutionLayer.InitKernelsAndBiases()");
            }
        }

        #endregion

        #region Test 2: Double-softmax bug

        /// <summary>
        /// When SoftMax layer + CategoricalCrossEntropy loss are used together,
        /// the SoftMax layer applies softmax in FeedForward, and then
        /// CategoricalCrossEntropy.Derivative applies softmax AGAIN internally.
        /// This means the network is computing softmax(softmax(logits)) - truth,
        /// which destroys gradients. PyTorch's CrossEntropyLoss takes raw logits.
        ///
        /// This test verifies whether double-softmax is happening.
        /// </summary>
        [TestMethod]
        public void SoftMaxPlusCrossEntropy_ShouldNotDoubleSoftmax()
        {
            // Simulate what happens in the CNN pipeline:
            // Dense layer outputs raw logits
            var logits = MatrixFactory.CreateColumnVector(new[] { 2.0f, 1.0f, 0.1f });

            // SoftMax layer applies softmax during FeedForward
            var softmaxLayer = new SoftMax(new InputOutputShape(1, 3, 1, 1), 3);
            var softmaxOutput = softmaxLayer.FeedFoward(new AnnTensor(null, logits));
            var afterSoftmax = softmaxOutput.ToColumnVector();

            // Verify softmax output sums to 1
            float sum = 0;
            for (int i = 0; i < afterSoftmax.Size; i++)
                sum += afterSoftmax[i];
            Assert.AreEqual(1.0f, sum, 0.001f, "Softmax output should sum to 1");

            // Now CategoricalCrossEntropy.Derivative receives the softmax output
            // but internally applies softmax AGAIN
            var loss = new CategoricalCrossEntropy();
            var truth = MatrixFactory.CreateColumnVector(new[] { 1.0f, 0.0f, 0.0f }); // class 0

            // What CategoricalCrossEntropy.Derivative does:
            // softmaxPred = SoftMax(predicted)  <-- applies softmax to already-softmaxed values!
            // return softmaxPred - truth
            var gradient = loss.Derivative(truth, afterSoftmax);

            // If there's NO double-softmax bug, gradient should be (softmax(logits) - truth)
            // afterSoftmax[0] ≈ 0.659, so gradient[0] should be ≈ 0.659 - 1.0 = -0.341
            var expectedSoftmax = CategoricalCrossEntropy.SoftMax(logits);
            float expectedGrad0 = expectedSoftmax[0] - truth[0];

            // What actually happens with double-softmax:
            // softmax(afterSoftmax) produces a FLATTENED distribution (values closer to 1/3)
            // so gradient[0] will be much closer to 0 than expected
            var doubleSoftmax = CategoricalCrossEntropy.SoftMax(afterSoftmax);

            Console.WriteLine($"Raw logits:           [{logits[0]:F4}, {logits[1]:F4}, {logits[2]:F4}]");
            Console.WriteLine($"After softmax layer:  [{afterSoftmax[0]:F4}, {afterSoftmax[1]:F4}, {afterSoftmax[2]:F4}]");
            Console.WriteLine($"After DOUBLE softmax: [{doubleSoftmax[0]:F4}, {doubleSoftmax[1]:F4}, {doubleSoftmax[2]:F4}]");
            Console.WriteLine($"Expected gradient[0]: {expectedGrad0:F4} (softmax(logits)[0] - 1.0)");
            Console.WriteLine($"Actual gradient[0]:   {gradient[0]:F4} (softmax(softmax(logits))[0] - 1.0)");
            Console.WriteLine();
            Console.WriteLine("If these differ significantly, you have a double-softmax bug.");
            Console.WriteLine("FIX: Either remove the SoftMax layer (let CategoricalCrossEntropy handle it),");
            Console.WriteLine("OR change CategoricalCrossEntropy.Derivative to not apply softmax internally.");

            // The gradient from double-softmax will be much smaller (flatter distribution)
            float actualGrad0 = gradient[0];

            // Assert they should be equal — if not, double-softmax is confirmed
            Assert.AreEqual(expectedGrad0, actualGrad0, 0.01f,
                $"DOUBLE SOFTMAX BUG: gradient[0] is {actualGrad0:F4} but should be {expectedGrad0:F4}. " +
                "CategoricalCrossEntropy.Derivative applies softmax to already-softmaxed values. " +
                "Fix: remove the SoftMax layer from the network, or change the loss function.");
        }

        #endregion

        #region Test 3: ReLU gradient correctness

        /// <summary>
        /// Verifies ReLU backprop correctly masks gradients.
        /// The ReLU derivative should be 1 where activation > 0, and 0 elsewhere.
        /// </summary>
        [TestMethod]
        public void ReLU_BackpropGradientMasking_IsCorrect()
        {
            var relu = new ReLUActivaction(new InputOutputShape(1, 4, 1, 1));

            // Forward pass: input with mix of positive and negative
            var input = MatrixFactory.CreateColumnVector(new[] { -1f, 2f, -3f, 4f });
            var activated = relu.Activate(input); // → [0, 2, 0, 4]

            Assert.AreEqual(0f, activated[0], 1e-6f);
            Assert.AreEqual(2f, activated[1], 1e-6f);
            Assert.AreEqual(0f, activated[2], 1e-6f);
            Assert.AreEqual(4f, activated[3], 1e-6f);

            // Backward pass: uniform incoming gradient
            var dE_dOut = MatrixFactory.CreateColumnVector(new[] { 1f, 1f, 1f, 1f });
            var dE_dIn = relu.BackPropagation(new AnnTensor(null, dE_dOut));
            var result = dE_dIn.ToColumnVector();

            // Expected: [0, 1, 0, 1] — gradient blocked where activation was 0
            Assert.AreEqual(0f, result[0], 1e-6f, "Gradient should be blocked where ReLU output was 0");
            Assert.AreEqual(1f, result[1], 1e-6f, "Gradient should pass through where ReLU output > 0");
            Assert.AreEqual(0f, result[2], 1e-6f, "Gradient should be blocked where ReLU output was 0");
            Assert.AreEqual(1f, result[3], 1e-6f, "Gradient should pass through where ReLU output > 0");
        }

        /// <summary>
        /// Verifies ReLU backprop works with matrix tensors (CNN path).
        /// </summary>
        [TestMethod]
        public void ReLU_MatrixBackprop_MasksCorrectly()
        {
            var relu = new ReLUActivaction(new InputOutputShape(2, 2, 1, 1));

            // Forward with a 2x2 matrix
            var mat = MatrixFactory.CreateMatrix(2, 2);
            mat[0, 0] = -1f; mat[0, 1] = 2f;
            mat[1, 0] = 3f;  mat[1, 1] = -4f;

            var inputTensor = new List<MatrixBase> { mat }.ToTensor();
            var output = relu.FeedFoward(inputTensor);

            // Verify forward: [0,2; 3,0]
            Assert.AreEqual(0f, output.Matrices[0][0, 0], 1e-6f);
            Assert.AreEqual(2f, output.Matrices[0][0, 1], 1e-6f);
            Assert.AreEqual(3f, output.Matrices[0][1, 0], 1e-6f);
            Assert.AreEqual(0f, output.Matrices[0][1, 1], 1e-6f);

            // Backward with uniform gradient
            var gradMat = MatrixFactory.CreateMatrix(2, 2);
            gradMat[0, 0] = 1f; gradMat[0, 1] = 1f;
            gradMat[1, 0] = 1f; gradMat[1, 1] = 1f;

            var gradTensor = new List<MatrixBase> { gradMat }.ToTensor();
            var backResult = relu.BackPropagation(gradTensor);

            // Expected: [0,1; 1,0]
            Assert.AreEqual(0f, backResult.Matrices[0][0, 0], 1e-6f);
            Assert.AreEqual(1f, backResult.Matrices[0][0, 1], 1e-6f);
            Assert.AreEqual(1f, backResult.Matrices[0][1, 0], 1e-6f);
            Assert.AreEqual(0f, backResult.Matrices[0][1, 1], 1e-6f);
        }

        #endregion

        #region Test 4: Gradient flow through full CNN chain

        /// <summary>
        /// Verifies that gradients flow all the way from loss back through
        /// Dense → Flatten → Pool → ReLU → Conv and that conv kernels receive non-zero gradients.
        /// </summary>
        [TestMethod]
        public void GradientFlow_ConvReceivesNonZeroGradients()
        {
            var inputShape = new InputOutputShape(8, 8, 1, 1);
            var conv = new ConvolutionLayer(inputShape, kernelCount: 2, kernelSquareDimension: 3, stride: 1);
            var relu = new ReLUActivaction(conv.OutputShape);
            var pool = new PoolingLayer(conv.OutputShape, stride: 2, kernelCount: 2,
                                         kernelSquareDimension: 2, kernelDepth: 1);
            var flatten = new FlattenLayer(pool.OutputShape, nodeCount: 1);
            var dense = new WeightedLayer(flatten.OutputShape, nodeCount: 2);

            var layers = new List<Layer> { conv, relu, pool, flatten, dense };
            var network = new GeneralFeedForwardANN(layers, 0.01f, 64, 2, new SquaredLoss());

            // Create input
            var img = MatrixFactory.CreateMatrix(8, 8);
            var rnd = new Random(42);
            for (int r = 0; r < 8; r++)
                for (int c = 0; c < 8; c++)
                    img[r, c] = (float)rnd.NextDouble();

            var inputTensor = new List<MatrixBase> { img }.ToTensor();
            var target = MatrixFactory.CreateColumnVector(new[] { 1f, 0f });

            // Reset accumulators
            foreach (var layer in layers)
                layer.ResetAccumulators();

            // Forward pass
            Tensor output = inputTensor;
            foreach (var layer in layers)
                output = layer.FeedFoward(output);

            // Backward pass
            var predicted = output.ToColumnVector();
            Assert.IsNotNull(predicted, "Forward pass should produce a column vector output");

            Tensor dE_dX = network.LossFunction.Derivative(target, predicted).ToTensor();
            foreach (var layer in layers.AsEnumerable().Reverse())
                dE_dX = layer.BackPropagation(dE_dX);

            // Check that conv kernel gradient accumulator has non-zero values
            bool hasNonZeroKernelGrad = false;
            for (int k = 0; k < conv.KernelGradientAccumulator.KernelCount; k++)
                for (int d = 0; d < conv.KernelGradientAccumulator.KernelDepth; d++)
                {
                    var gradKernel = conv.KernelGradientAccumulator[k, d];
                    for (int r = 0; r < gradKernel.Rows; r++)
                        for (int c = 0; c < gradKernel.Cols; c++)
                            if (System.Math.Abs(gradKernel[r, c]) > 1e-10f)
                                hasNonZeroKernelGrad = true;
                }

            Assert.IsTrue(hasNonZeroKernelGrad,
                "Conv kernel gradient accumulator should have non-zero values after backprop. " +
                "If all zeros, gradients are not flowing back to the conv layer.");

            // Check bias gradients too
            bool hasNonZeroBiasGrad = false;
            for (int k = 0; k < conv.BiasesGradientAccumulator.Count; k++)
            {
                var biasGrad = conv.BiasesGradientAccumulator[k];
                for (int r = 0; r < biasGrad.Rows; r++)
                    for (int c = 0; c < biasGrad.Cols; c++)
                        if (System.Math.Abs(biasGrad[r, c]) > 1e-10f)
                            hasNonZeroBiasGrad = true;
            }

            Assert.IsTrue(hasNonZeroBiasGrad,
                "Conv bias gradient accumulator should have non-zero values after backprop.");

            // Check dense layer got gradients
            Assert.IsNotNull(dense.LastWeightGradient, "Dense weight gradient should not be null");
            Assert.IsNotNull(dense.LastBiasGradient, "Dense bias gradient should not be null");

            Console.WriteLine("Gradient flow verified: all layers received non-zero gradients.");
        }

        #endregion

        #region Test 5: SoftMax + CrossEntropy gradient integration

        /// <summary>
        /// Verifies CategoricalCrossEntropy.Derivative computes softmax(logits) - truth.
        /// When passed raw logits (not already-softmaxed values), this is correct.
        /// </summary>
        [TestMethod]
        public void CategoricalCrossEntropy_DerivativeWithRawLogits_IsCorrect()
        {
            var loss = new CategoricalCrossEntropy();

            // Raw logits from dense layer
            var logits = MatrixFactory.CreateColumnVector(new[] { 1.0f, 2.0f, 3.0f });
            var truth = MatrixFactory.CreateColumnVector(new[] { 1.0f, 0.0f, 0.0f }); // class 0

            // Expected: softmax([1,2,3]) - [1,0,0]
            var expectedSoftmax = CategoricalCrossEntropy.SoftMax(logits);
            float expectedGrad0 = expectedSoftmax[0] - 1.0f; // ≈ 0.0900 - 1.0 = -0.910
            float expectedGrad1 = expectedSoftmax[1] - 0.0f; // ≈ 0.2447
            float expectedGrad2 = expectedSoftmax[2] - 0.0f; // ≈ 0.6652

            var gradient = loss.Derivative(truth, logits);

            Console.WriteLine($"Softmax([1,2,3]) = [{expectedSoftmax[0]:F4}, {expectedSoftmax[1]:F4}, {expectedSoftmax[2]:F4}]");
            Console.WriteLine($"Expected gradient: [{expectedGrad0:F4}, {expectedGrad1:F4}, {expectedGrad2:F4}]");
            Console.WriteLine($"Actual gradient:   [{gradient[0]:F4}, {gradient[1]:F4}, {gradient[2]:F4}]");

            Assert.AreEqual(expectedGrad0, gradient[0], 0.001f, "Gradient[0] mismatch");
            Assert.AreEqual(expectedGrad1, gradient[1], 0.001f, "Gradient[1] mismatch");
            Assert.AreEqual(expectedGrad2, gradient[2], 0.001f, "Gradient[2] mismatch");

            // Verify gradient sums to 0 (property of softmax - one-hot)
            float gradSum = gradient[0] + gradient[1] + gradient[2];
            Assert.AreEqual(0f, gradSum, 0.001f,
                "Gradient of softmax-crossentropy should sum to 0");
        }

        #endregion

        #region Test 6: Conv weight update proportional to learning rate

        /// <summary>
        /// Verifies that larger learning rates produce proportionally larger weight updates.
        /// If this fails, the gradient scaling or accumulation is wrong.
        /// </summary>
        [TestMethod]
        public void ConvWeightUpdate_ProportionalToLearningRate()
        {
            // Train with two different learning rates and compare weight deltas
            float delta1 = TrainOneStepAndGetConvWeightDelta(0.001f);
            float delta2 = TrainOneStepAndGetConvWeightDelta(0.01f);

            Console.WriteLine($"Weight delta with lr=0.001: {delta1:E4}");
            Console.WriteLine($"Weight delta with lr=0.01:  {delta2:E4}");
            Console.WriteLine($"Ratio: {delta2 / delta1:F1}x (expected ~10x)");

            Assert.IsTrue(delta2 > delta1,
                $"Higher LR should cause larger weight updates. Got lr=0.001→{delta1:E4}, lr=0.01→{delta2:E4}");

            float ratio = delta2 / delta1;
            Assert.IsTrue(ratio > 5f && ratio < 15f,
                $"Weight update ratio should be ~10x (got {ratio:F1}x). " +
                "If far off, gradient scaling or accumulation count is wrong.");
        }

        private float TrainOneStepAndGetConvWeightDelta(float learningRate)
        {
            var inputShape = new InputOutputShape(8, 8, 1, 1);
            var conv = new ConvolutionLayer(inputShape, kernelCount: 1, kernelSquareDimension: 3, stride: 1);
            var relu = new ReLUActivaction(conv.OutputShape);
            var pool = new PoolingLayer(conv.OutputShape, stride: 2, kernelCount: 1,
                                         kernelSquareDimension: 2, kernelDepth: 1);
            var flatten = new FlattenLayer(pool.OutputShape, nodeCount: 1);
            var dense = new WeightedLayer(flatten.OutputShape, nodeCount: 2);

            var layers = new List<Layer> { conv, relu, pool, flatten, dense };

            // Record initial kernel value
            float kernelBefore = conv.Kernels[0, 0][0, 0];

            // Forward
            var img = MatrixFactory.CreateMatrix(8, 8);
            var rnd = new Random(42);
            for (int r = 0; r < 8; r++)
                for (int c = 0; c < 8; c++)
                    img[r, c] = (float)rnd.NextDouble();

            var inputTensor = new List<MatrixBase> { img }.ToTensor();
            var target = MatrixFactory.CreateColumnVector(new[] { 1f, 0f });

            foreach (var layer in layers)
                layer.ResetAccumulators();

            Tensor output = inputTensor;
            foreach (var layer in layers)
                output = layer.FeedFoward(output);

            // Backward
            var loss = new SquaredLoss();
            Tensor dE_dX = loss.Derivative(target, output.ToColumnVector()).ToTensor();
            foreach (var layer in layers.AsEnumerable().Reverse())
                dE_dX = layer.BackPropagation(dE_dX);

            // Update weights
            foreach (var layer in layers)
                layer.UpdateWeightsAndBiasesWithScaledGradients(learningRate);

            float kernelAfter = conv.Kernels[0, 0][0, 0];
            return System.Math.Abs(kernelAfter - kernelBefore);
        }

        #endregion

        #region Test 7: End-to-end CNN convergence on simple 2D data

        /// <summary>
        /// Tests CNN convergence on trivially separable 2D data.
        /// Top-half-bright images → class 0, bottom-half-bright → class 1.
        /// If the CNN can't learn this, there's a fundamental gradient or update bug.
        ///
        /// Uses Dense → SquaredLoss (no SoftMax) to avoid the double-softmax issue.
        /// </summary>
        [TestMethod]
        public void CNN_ConvergesOnSimple2DClassification_NoSoftmax()
        {
            var trainingPairs = CreateSimple2DData();
            var inputShape = new InputOutputShape(8, 8, 1, 1);

            var conv = new ConvolutionLayer(inputShape, kernelCount: 4, kernelSquareDimension: 3, stride: 1);
            var relu = new ReLUActivaction(conv.OutputShape);
            var pool = new PoolingLayer(conv.OutputShape, stride: 2, kernelCount: 4,
                                         kernelSquareDimension: 2, kernelDepth: 1);
            var flatten = new FlattenLayer(pool.OutputShape, nodeCount: 1);
            var dense = new WeightedLayer(flatten.OutputShape, nodeCount: 2);
            var sigmoid = new SigmoidActivation(dense.OutputShape, nodeCount: 2);

            var layers = new List<Layer> { conv, relu, pool, flatten, dense, sigmoid };
            var network = new GeneralFeedForwardANN(layers, 0.01f, 64, 2, new SquaredLoss());

            float initialLoss = CalculateAvgLoss(network, layers, trainingPairs);
            Console.WriteLine($"Initial avg loss: {initialLoss:F6}");

            // Train for 20 epochs manually (no batch infrastructure needed for this small dataset)
            for (int epoch = 0; epoch < 20; epoch++)
            {
                foreach (var pair in trainingPairs)
                {
                    foreach (var layer in layers)
                        layer.ResetAccumulators();

                    Tensor output = pair.Input;
                    foreach (var layer in layers)
                        output = layer.FeedFoward(output);

                    Tensor dE_dX = network.LossFunction.Derivative(
                        pair.Output.ToColumnVector(), output.ToColumnVector()).ToTensor();
                    foreach (var layer in layers.AsEnumerable().Reverse())
                        dE_dX = layer.BackPropagation(dE_dX);

                    foreach (var layer in layers)
                        layer.UpdateWeightsAndBiasesWithScaledGradients(network.LearningRate);
                }

                if (epoch % 5 == 0)
                {
                    float loss = CalculateAvgLoss(network, layers, trainingPairs);
                    Console.WriteLine($"Epoch {epoch}: avg loss = {loss:F6}");
                }
            }

            float finalLoss = CalculateAvgLoss(network, layers, trainingPairs);
            Console.WriteLine($"Final avg loss: {finalLoss:F6}");

            Assert.IsTrue(finalLoss < initialLoss,
                $"Loss should decrease on trivially separable data. " +
                $"Initial: {initialLoss:F6}, Final: {finalLoss:F6}. " +
                "If loss didn't decrease, there's a fundamental gradient or weight update bug.");
        }

        /// <summary>
        /// Same test but with SoftMax + CategoricalCrossEntropy to detect the double-softmax problem.
        /// If Test 7a passes but this fails, the double-softmax bug is confirmed.
        /// </summary>
        [TestMethod]
        public void CNN_ConvergesOnSimple2DClassification_WithSoftmax()
        {
            var trainingPairs = CreateSimple2DData();
            var inputShape = new InputOutputShape(8, 8, 1, 1);

            var conv = new ConvolutionLayer(inputShape, kernelCount: 4, kernelSquareDimension: 3, stride: 1);
            var relu = new ReLUActivaction(conv.OutputShape);
            var pool = new PoolingLayer(conv.OutputShape, stride: 2, kernelCount: 4,
                                         kernelSquareDimension: 2, kernelDepth: 1);
            var flatten = new FlattenLayer(pool.OutputShape, nodeCount: 1);
            var dense = new WeightedLayer(flatten.OutputShape, nodeCount: 2);
            var softmax = new SoftMax(dense.OutputShape, nodeCount: 2);

            var layers = new List<Layer> { conv, relu, pool, flatten, dense, softmax };
            var network = new GeneralFeedForwardANN(layers, 0.01f, 64, 2, new CategoricalCrossEntropy());

            float initialLoss = CalculateAvgLoss(network, layers, trainingPairs);
            Console.WriteLine($"Initial avg loss: {initialLoss:F6}");

            for (int epoch = 0; epoch < 20; epoch++)
            {
                foreach (var pair in trainingPairs)
                {
                    foreach (var layer in layers)
                        layer.ResetAccumulators();

                    Tensor output = pair.Input;
                    foreach (var layer in layers)
                        output = layer.FeedFoward(output);

                    Tensor dE_dX = network.LossFunction.Derivative(
                        pair.Output.ToColumnVector(), output.ToColumnVector()).ToTensor();
                    foreach (var layer in layers.AsEnumerable().Reverse())
                        dE_dX = layer.BackPropagation(dE_dX);

                    foreach (var layer in layers)
                        layer.UpdateWeightsAndBiasesWithScaledGradients(network.LearningRate);
                }

                if (epoch % 5 == 0)
                {
                    float loss = CalculateAvgLoss(network, layers, trainingPairs);
                    Console.WriteLine($"Epoch {epoch}: avg loss = {loss:F6}");
                }
            }

            float finalLoss = CalculateAvgLoss(network, layers, trainingPairs);
            Console.WriteLine($"Final avg loss: {finalLoss:F6}");

            Assert.IsTrue(finalLoss < initialLoss,
                $"Loss should decrease. Initial: {initialLoss:F6}, Final: {finalLoss:F6}. " +
                "If the NoSoftmax variant passes but this fails, you have the DOUBLE SOFTMAX bug: " +
                "SoftMax layer applies softmax, then CategoricalCrossEntropy.Derivative applies it again.");
        }

        #endregion

        #region Test 8: Batch training vs manual per-sample training equivalence

        /// <summary>
        /// Verifies that ConvolutionRenderContext.BatchTrain with batch_size=N
        /// produces the same weight updates as manually processing N samples.
        /// If these diverge, the batch accumulation or averaging is wrong.
        /// </summary>
        [TestMethod]
        public void CNNBatchTrain_MatchesManualTraining()
        {
            var trainingPairs = CreateSimple2DData();
            int batchSize = trainingPairs.Count; // use all samples as one batch

            // Network 1: trained via manual loop
            var (conv1, layers1, network1) = BuildSmallCNN(0.01f);

            // Record initial weights
            float initialKernel = conv1.Kernels[0, 0][0, 0];
            float initialDenseW = (layers1[4] as WeightedLayer).Weights[0, 0];

            foreach (var layer in layers1)
                layer.ResetAccumulators();

            foreach (var pair in trainingPairs)
            {
                Tensor output = pair.Input;
                foreach (var layer in layers1)
                    output = layer.FeedFoward(output);

                Tensor dE_dX = network1.LossFunction.Derivative(
                    pair.Output.ToColumnVector(), output.ToColumnVector()).ToTensor();
                foreach (var layer in layers1.AsEnumerable().Reverse())
                    dE_dX = layer.BackPropagation(dE_dX);
            }

            foreach (var layer in layers1)
                layer.UpdateWeightsAndBiasesWithScaledGradients(network1.LearningRate);

            float manualKernel = conv1.Kernels[0, 0][0, 0];
            float manualDenseW = (layers1[4] as WeightedLayer).Weights[0, 0];

            // Network 2: trained via ConvolutionRenderContext
            var (conv2, layers2, network2) = BuildSmallCNN(0.01f);

            var mockSet = new MockTrainingSet2D(trainingPairs,
                new InputOutputShape(8, 8, 1, 1), new InputOutputShape(1, 2, 1, 1));
            var ctx = new ConvolutionRenderContext(network2, batchSize, mockSet);
            ctx.EpochTrain(1);

            float batchKernel = conv2.Kernels[0, 0][0, 0];
            float batchDenseW = (layers2[4] as WeightedLayer).Weights[0, 0];

            Console.WriteLine($"Conv kernel[0,0][0,0]: manual={manualKernel:F6}, batch={batchKernel:F6}");
            Console.WriteLine($"Dense weight[0,0]:     manual={manualDenseW:F6}, batch={batchDenseW:F6}");

            Assert.AreEqual(manualKernel, batchKernel, 1e-4f,
                "Conv kernel should match between manual and batch training");
            Assert.AreEqual(manualDenseW, batchDenseW, 1e-4f,
                "Dense weights should match between manual and batch training");
        }

        #endregion

        #region Helpers

        private static List<TrainingPair> CreateSimple2DData()
        {
            var pairs = new List<TrainingPair>();

            // 10 samples: top-bright = class 0, bottom-bright = class 1
            for (int s = 0; s < 10; s++)
            {
                var img = MatrixFactory.CreateMatrix(8, 8);
                int label = s < 5 ? 0 : 1;

                for (int r = 0; r < 8; r++)
                    for (int c = 0; c < 8; c++)
                    {
                        if (label == 0)
                            img[r, c] = r < 4 ? 0.8f : 0.1f; // bright top
                        else
                            img[r, c] = r >= 4 ? 0.8f : 0.1f; // bright bottom
                    }

                var output = MatrixFactory.CreateColumnVector(
                    label == 0 ? new[] { 1f, 0f } : new[] { 0f, 1f });
                pairs.Add(new TrainingPair(
                    new List<MatrixBase> { img }.ToTensor(),
                    new AnnTensor(null, output)));
            }

            return pairs;
        }

        private static (ConvolutionLayer conv, List<Layer> layers, GeneralFeedForwardANN network)
            BuildSmallCNN(float lr)
        {
            var inputShape = new InputOutputShape(8, 8, 1, 1);
            var conv = new ConvolutionLayer(inputShape, kernelCount: 2, kernelSquareDimension: 3, stride: 1);
            var relu = new ReLUActivaction(conv.OutputShape);
            var pool = new PoolingLayer(conv.OutputShape, stride: 2, kernelCount: 2,
                                         kernelSquareDimension: 2, kernelDepth: 1);
            var flatten = new FlattenLayer(pool.OutputShape, nodeCount: 1);
            var dense = new WeightedLayer(flatten.OutputShape, nodeCount: 2);

            var layers = new List<Layer> { conv, relu, pool, flatten, dense };
            var network = new GeneralFeedForwardANN(layers, lr, 64, 2, new SquaredLoss());

            return (conv, layers, network);
        }

        private static float CalculateAvgLoss(GeneralFeedForwardANN network, List<Layer> layers,
            List<TrainingPair> pairs)
        {
            float total = 0;
            foreach (var pair in pairs)
            {
                Tensor output = pair.Input;
                foreach (var layer in layers)
                    output = layer.FeedFoward(output);
                total += network.GetTotallLoss(pair, output.ToColumnVector());
            }
            return total / pairs.Count;
        }

        /// <summary>
        /// Mock training set that returns 2D image data (not flattened).
        /// </summary>
        private class MockTrainingSet2D : ITrainingSet
        {
            private readonly List<TrainingPair> _pairs;
            private readonly InputOutputShape _inputShape;
            private readonly InputOutputShape _outputShape;

            public MockTrainingSet2D(List<TrainingPair> pairs,
                InputOutputShape inputShape, InputOutputShape outputShape)
            {
                _pairs = pairs;
                _inputShape = inputShape;
                _outputShape = outputShape;
            }

            public int Width => _inputShape.Width;
            public int Height => _inputShape.Height;
            public int Depth => _inputShape.Depth;
            public InputOutputShape OutputShape => _outputShape;
            public int NumClasses => _outputShape.TotalFlattenedSize;
            public int NumberOfSamples => _pairs.Count;
            public int NumberOfLabels => _pairs.Count;
            public List<TrainingPair> TrainingList => _pairs;

            public List<TrainingPair> BuildNewRandomizedTrainingList(bool normalized2D)
            {
                return new List<TrainingPair>(_pairs);
            }
        }

        #endregion
    }
}
