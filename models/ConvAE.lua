local nn = require 'nn'
require 'cunn'

local Model = {}

function Model:createAutoencoder(X, feature_size)

  -- Create encoder
  self.encoder = nn.Sequential():cuda()
  self.encoder:add(nn.View(3, 32, 32):cuda())
  self.encoder:add(nn.SpatialConvolutionMM(3, 128, 5, 5, 1, 1):cuda())
  self.encoder:add(nn.ReLU():cuda())
  local pool1 = nn.SpatialMaxPooling(2, 2, 2, 2):cuda()
  self.encoder:add(pool1)
  self.encoder:add(nn.SpatialConvolutionMM(128, 256, 5, 5, 1, 1):cuda())
  self.encoder:add(nn.ReLU():cuda())
  local pool2 = nn.SpatialMaxPooling(2, 2, 2, 2):cuda()
  self.encoder:add(pool2)
  self.encoder:add(nn.SpatialConvolutionMM(256, 512, 4, 4, 1, 1):cuda())
  self.encoder:add(nn.ReLU():cuda())
  self.encoder:add(nn.SpatialConvolutionMM(512, 1024, 2, 2, 1, 1):cuda())
  self.encoder:add(nn.ReLU():cuda())
  self.encoder:add(nn.Dropout(0.2):cuda())
  self.encoder:add(nn.SpatialConvolutionMM(1024, feature_size, 1, 1, 1, 1):cuda())
  self.encoder:add(nn.Reshape(feature_size):cuda())

  -- Create decoder
  self.decoder = nn.Sequential():cuda()
  self.decoder:add(nn.View(feature_size, 1, 1):cuda())
  self.decoder:add(nn.SpatialFullConvolution(feature_size, 1024, 1, 1, 1, 1):cuda())
  self.decoder:add(nn.ReLU():cuda())
  self.decoder:add(nn.SpatialFullConvolution(1024, 512, 2, 2, 1, 1):cuda())
  self.decoder:add(nn.ReLU():cuda())
  self.decoder:add(nn.SpatialFullConvolution(512, 256, 4, 4, 1, 1):cuda())
  self.decoder:add(nn.SpatialMaxUnpooling(pool2):cuda())
  self.decoder:add(nn.ReLU():cuda())
  self.decoder:add(nn.SpatialFullConvolution(256, 128, 5, 5, 1, 1):cuda())
  self.decoder:add(nn.SpatialMaxUnpooling(pool1):cuda())
  self.decoder:add(nn.ReLU():cuda())
  self.decoder:add(nn.SpatialFullConvolution(128, 1, 5, 5, 1, 1):cuda())
  self.decoder:add(nn.View(1024):cuda())
  self.decoder:add(nn.Sigmoid():cuda())

  -- Create autoencoder
  self.autoencoder = nn.Sequential():cuda()
  self.autoencoder:add(self.encoder)
  self.autoencoder:add(self.decoder)
end

return Model
