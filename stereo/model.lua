-- siamese network 
--
-- David-Alexandre Beaupre
--

require 'nn'
require 'cunn'

function create_model(max_dips, nChannel)
	-- whole network
	local model = nn.Sequential()
	-- base = combination of left and right network
	local base = nn.ParallelTable()
	local left = nn.Sequential()

	-- building block (convolution, batch normalization and ReLU activation)
	local function ConvBNReLU(nInputPlane, nOutputPlane, kw, kh)
      left:add(nn.SpatialConvolution(nInputPlane, nOutputPlane, kw, kh))
	  left:add(nn.SpatialBatchNormalization(nOutputPlane, 1e-3))
	  left:add(nn.ReLU(true))
	  return left
	end

	local c = nChannel or 3
	-- 5 layers for feature extraction
	ConvBNReLU(c, 32, 7, 7):add(nn.Dropout(0.2))
	ConvBNReLU(32, 32, 7, 7):add(nn.Dropout(0.5))
	ConvBNReLU(32, 64, 7, 7):add(nn.Dropout(0.5))
	ConvBNReLU(64, 64, 7, 7):add(nn.Dropout(0.5))
	ConvBNReLU(64, 64, 7, 7):add(nn.Dropout(0.5))
	-- no ReLU activation for last layer
    left:add(nn.SpatialConvolution(64, 64, 7, 7))
	left:add(nn.SpatialBatchNormalization(64, 1e-3))

	-- left and right share parameters
	local right = left:clone('weight', 'bias', 'gradWeight', 'gradBias')
	left:add(nn.Transpose({2, 3}, {3, 4}))
	-- left is a vector
	left:add(nn.Reshape(1, 64))
	-- right is a volume
	right:add(nn.Reshape(64, max_dips))
	base:add(left):add(right)
	model:add(base)
	-- correlation between left and right (for every disparity location)
	model:add(nn.MM())
	model:add(nn.Reshape(max_dips))
	-- numerically stable probabilities
	model:add(nn.LogSoftMax())

	return model
end