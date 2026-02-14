using MatrixLibrary;
using MatrixLibrary.BaseClasses;
using System.Diagnostics;

namespace NeuralNets
{
    /*
     * This is a classic topic, but one where the details often trip people up. Because Softmax is a vector-to-vector function (unlike ReLU or Sigmoid which are element-wise), its derivative is a **Jacobian matrix**, not a simple vector.

Here is the breakdown of the Feed Forward, Derivative, and Backpropagation methods for a Softmax layer.

---

### 1. Feed Forward

In the forward pass, you take the raw output scores (logits) from the previous layer, usually denoted as vectors , and convert them into a probability distribution.

**The Formula**
For a vector  of length , the Softmax value for the -th element is:

Where:

*  is the input vector (logits).
*  is the output vector (probabilities summing to 1).

> **Expert Tip (Numerical Stability):** When implementing this in code (e.g., Python/NumPy), raw exponentials can explode (overflow) if  is large.
> A standard trick is to subtract the maximum value of  from the entire vector before applying the exponential.
> 
> 
> 
> This yields the exact same mathematical result but prevents `NaN` errors.

---

### 2. The Derivative (The Jacobian)

This is the complex part. Because the output  depends on the sum of **all** inputs in the denominator, changing one input  affects **every** output element.

Therefore, we cannot just calculate a single derivative vector. We calculate the partial derivative of every output with respect to every input: .

There are two cases to solve for:

**Case 1:  (Derivative with respect to its own logit)**
Using the quotient rule, the derivative simplifies to:


**Case 2:  (Derivative with respect to other logits)**
The derivative is:


**Combined (Jacobian Matrix)**
We can combine these using the Kronecker delta ( if , else ):

This results in a square Jacobian matrix where diagonal elements are  and off-diagonal elements are .

---

### 3. Backpropagation

In backpropagation, you receive a gradient vector from the subsequent layer (usually the loss function), denoted as  (or ). You need to pass this gradient backward to  to get  (or ).

#### Scenario A: The "Hard" Way (Softmax Isolated)

If you are treating Softmax as a standalone layer (rare but possible), you perform a vector-matrix multiplication using the Jacobian we derived above.

In code, this is computationally expensive because you are computing a full matrix multiplication for every sample in your batch.

#### Scenario B: The "Standard" Way (Softmax + Cross-Entropy)

In 99% of classification tasks, Softmax is the output layer and is paired immediately with **Categorical Cross-Entropy Loss**.

When you combine the derivative of Cross-Entropy with the derivative of Softmax, the math cancels out beautifully.

* **Loss Gradient:**  (where  is the one-hot encoded true label).
* **Softmax Gradient:** The Jacobian described above.

If you multiply these, the final gradient with respect to the logits  becomes incredibly simple:

Or in vector form:


This is simply **(Predicted Probability - True Label)**.

### Summary of Methods

| Method | Mathematical Operation | Complexity |
| --- | --- | --- |
| **Feed Forward** |  |  |
| **Derivative** | Jacobian Matrix:  |  |
| **Backprop (Combined)** |  |  |

**Would you like to see a NumPy implementation of the stable Feed Forward and Backprop step?**
**/

    public class SoftMax : Layer, IActivationFunction
    {
        public SoftMax(InputOutputShape inputShape, int nodeCount, int randomSeed = 55) : base(inputShape, nodeCount, randomSeed)
        {
        }

        public Tensor LastActivation {  get; private set; }

        public override InputOutputShape OutputShape => throw new NotImplementedException();

        public ColumnVectorBase Activate(ColumnVectorBase input)
        {
            float max = input.GetMax();
            float scaleFactor = SumExpEMinusMax(max, input);
            float[] softMaxVec = new float[input.Size];
            for (int i = 0; i < input.Size; i++)
                softMaxVec[i] = (float)(Math.Exp(input[i] - max) / scaleFactor);

            ColumnVectorBase la = MatrixFactory.CreateColumnVector(softMaxVec);
            LastActivation = la.ToTensor();
            return la;
        }

        public MatrixBase Activate(MatrixBase input)
        {
            throw new NotImplementedException();
        }

        public List<MatrixBase> Activate(List<MatrixBase> input)
        {
            throw new NotImplementedException();
        }

        // https://www.mldawn.com/the-derivative-of-softmaxz-function-w-r-t-z/
        // https://stats.stackexchange.com/questions/453539/softmax-derivative-implementation
        public ColumnVectorBase Derivative(ColumnVectorBase lastActivation)
        {
            Debug.Assert(lastActivation != null);
            throw new InvalidOperationException("Don't call derivative on softmax, just use the softmax*crossentropy derivative which is a-y'");
        }
        private static float SumExpEMinusMax(float max, ColumnVectorBase vec)
        {
            float scale = 0;
            for (int i = 0; i < vec.Size; i++)
            {
                scale += (float)Math.Exp(vec[i] - max);
            }
            return scale;
        }

        public override Tensor FeedFoward(Tensor input)
        {
            throw new NotImplementedException();
        }

        public override Tensor BackPropagation(Tensor dE_dY)
        {
            throw new NotImplementedException();
        }
    }
}