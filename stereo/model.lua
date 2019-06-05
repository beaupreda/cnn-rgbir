-- siamese network 
--
-- David-Alexandre Beaupre
--

require 'nn'
require 'cunn'

function create_model(max_dips, nChannel)
	local m = nn.Sequential()

	local bottom = nn.ParallelTable()
	local bottom_left = nn.Sequential()
	-- building block
	local function ConvBNReLU(nInputPlane, nOutputPlane, kw, kh)
      bottom_left:add(nn.SpatialConvolution(nInputPlane, nOutputPlane, kw, kh))
	  bottom_left:add(nn.SpatialBatchNormalization(nOutputPlane,1e-3))
	  bottom_left:add(nn.ReLU(true))
	  return bottom_left
	end

	local c = nChannel or 3
	ConvBNReLU(c,32, 7,7):add(nn.Dropout(0.2))
	ConvBNReLU(32,32,7,7):add(nn.Dropout(0.5))
	ConvBNReLU(32,64,7,7):add(nn.Dropout(0.5))
	ConvBNReLU(64,64,7,7):add(nn.Dropout(0.5))
	ConvBNReLU(64,64,7,7):add(nn.Dropout(0.5))
    bottom_left:add(nn.SpatialConvolution(64, 64, 7, 7))
	bottom_left:add(nn.SpatialBatchNormalization(64,1e-3))

	local bottom_right = bottom_left:clone('weight','bias','gradWeight','gradBias')
    bottom_left:add(nn.Transpose({2,3}, {3,4}))
	bottom_left:add(nn.Reshape(1, 64))

	bottom_right:add(nn.Reshape(64,max_dips))

	bottom:add(bottom_left):add(bottom_right)

	m:add(bottom)
	m:add(nn.MM())
	m:add(nn.Reshape(max_dips))

	m:add(nn.LogSoftMax())

	return m

end