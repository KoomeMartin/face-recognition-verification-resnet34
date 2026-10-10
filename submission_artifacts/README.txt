- **Model**: ResNet-34 implemented from scratch (BasicBlock x [3,4,6,3]) adapted for
  112x112 faces. Two deliberate departures from the ImageNet ResNet: a 3x3 stride-1 stem
  with no maxpool (the standard 7x7/s2 + maxpool stem downsamples 4x before the first
  residual block, destroying the fine periocular and nasolabial texture that carries
  identity at this resolution), and stage strides 2/2/2/2 so the pyramid is
  112->56->28->14->7, preserving the same 32x total reduction. The last BatchNorm gamma
  in every residual branch is zero-initialised, so each block starts as an identity map.
  The embedding head is a global depthwise convolution (7x7, groups=512, MobileFaceNet
  style) rather than global average pooling: GAP weights all 49 spatial positions
  equally, but these faces are aligned so position is informative. Dropout(0.3) ->
  Linear(512, no bias) -> BatchNorm1d yields the 512-d face embedding shared by both
  tasks. ~26M parameters, under the 30M budget. No pretrained weights.
- **Training Strategy**: SGD (momentum 0.9, Nesterov, weight decay 5e-4 applied to
  conv/linear weights only -- BatchNorm gammas and biases exempt), 2-epoch linear warm-up
  then cosine decay to 1e-5, batch size 256, mixed-precision training with gradient
  clipping at 3.0. Loss: softmax cross-entropy with 0.1 label smoothing on a linear
  classification head. Verification is performed by transfer: the classification head is
  discarded and pairs are scored by cosine similarity between L2-normalised embeddings,
  each image passed separately through the shared encoder. Checkpoints are selected on
  0.5*val_accuracy + 0.5*(100 - val_EER) to match the Kaggle metric rather than on
  classification accuracy alone.
- **Augmentations**: random resized crop (scale 0.75-1.0), horizontal flip, colour jitter
  (p=0.5), random grayscale (p=0.05), rotation +/-10 degrees (p=0.3), random erasing
  (p=0.25). No vertical flips and no large rotations -- upside-down faces do not occur at
  test time.
- **Results**: <FILL IN> epochs, best validation combined score 96.72%
  (classification 95.69%, verification EER 2.24%).
- **Notebook Execution**: set config['data_root'] for your platform, run top to bottom.