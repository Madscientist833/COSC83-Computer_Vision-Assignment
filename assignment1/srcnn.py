import torch
import torch.nn as nn
import torch.nn.functional as F
import math

#5%
class ResidualBlock(nn.Module):
    """Residual block with skip connection"""
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        # TODO: Implement the residual block constructor
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(channels)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)   
        self.bn2   = nn.BatchNorm2d(channels)
        
    def forward(self, x):
        # TODO: Implement the forward pass of the residual block
        # 1. Store the input as the residual
        # 2. Pass the input through the first conv -> batch norm -> ReLU sequence
        # 3. Pass the result through the second conv -> batch norm sequence
        # 4. Add the residual to implement the skip connection
        # 5. Apply ReLU and return the result
        residual = x 
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + residual
        out = self.relu(out)
        return out

#5%
class UpscaleBlock(nn.Module):
    """Upscale block using sub-pixel convolution"""
    def __init__(self, in_channels, scale_factor):
        super(UpscaleBlock, self).__init__()
        # TODO: Implement the upscale block constructor
        # 1. Calculate output channels for sub-pixel convolution (hint: multiply in_channels by scale_factor^2)
        # 2. Create a convolutional layer with kernel size 3 and padding 1
        # 3. Create a pixel shuffle layer with the given scale factor
        # 4. Create a ReLU activation
        out_channels = in_channels * (scale_factor ** 2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(scale_factor)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        # TODO: Implement the forward pass of the upscale block
        # 1. Apply the convolutional layer
        # 2. Apply the pixel shuffle operation
        # 3. Apply ReLU and return the result
        out = self.conv(x)
        out = self.pixel_shuffle(out)
        out = self.relu(out)
        return out

#10%
class SuperResolutionCNN(nn.Module):
    def __init__(self, scale_factor=4, num_channels=3, num_features=64, num_blocks=16):
        """
        SuperResolution CNN with residual blocks and sub-pixel convolution
        
        Args:
            scale_factor (int): Upscaling factor
            num_channels (int): Number of input/output channels
            num_features (int): Number of feature channels
            num_blocks (int): Number of residual blocks
        """
        super(SuperResolutionCNN, self).__init__()
        self.scale_factor = scale_factor
        
        # TODO: Implement the constructor for the Super Resolution CNN
        # 1. Create an initial convolution layer with kernel size 9, padding 4, followed by ReLU
        # 2. Create a sequence of residual blocks (use the ResidualBlock class)
        # 3. Create a mid convolution layer with kernel size 3, padding 1, followed by batch norm
        # 4. Create upscaling layers based on the scale factor:
        #    - For scale factors 2, 4, and 8 (powers of 2), use multiple x2 upscaling blocks
        #    - For scale factor 3, use a single x3 upscaling block
        #    - Raise an error for other scale factors
        # 5. Create a final convolution layer with kernel size 9, padding 4
        # 6. Initialize the weights using the _initialize_weights method
        
        self.initial_conv = nn.Sequential(
            nn.Conv2d(num_channels, num_features, kernel_size=9, padding=4),
            nn.ReLU(inplace=True)
        )
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(num_features) for _ in range(num_blocks)]
        )
        self.mid_conv = nn.Sequential(
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features)
        )
        if scale_factor in [2, 4, 8]:
            upscaling_blocks = []
            for _ in range(int(math.log2(scale_factor))):
                upscaling_blocks.append(UpscaleBlock(num_features, 2))
            self.upscaling = nn.Sequential(*upscaling_blocks)
        elif scale_factor == 3:
            self.upscaling = UpscaleBlock(num_features, 3)
        else:
            raise ValueError("Unsupported scale factor")
        self.final_conv = nn.Conv2d(num_features, num_channels, kernel_size=9, padding=4)
        self._initialize_weights()

        
    def _initialize_weights(self):
        # TODO: Implement weight initialization
        # For each module in the model:
        # 1. For convolutional layers, use Kaiming normal initialization for weights and zero initialization for biases
        # 2. For batch normalization layers, use ones for weights and zeros for biases
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        
    def forward(self, x):
        # TODO: Implement the forward pass of the Super Resolution CNN
        # 1. Apply the initial convolution and store the output for the global skip connection
        # 2. Pass the features through the residual blocks
        # 3. Apply the mid convolution
        # 4. Add the initial features (global residual learning)
        # 5. Apply the upscaling layers
        # 6. Apply the final convolution and return the result
        initial_features = self.initial_conv(x)
        residual = self.residual_blocks(initial_features)
        mid = self.mid_conv(residual)
        out = initial_features + mid
        out = self.upscaling(out)
        out = self.final_conv(out)
        return out