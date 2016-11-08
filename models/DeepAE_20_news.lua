local nn = require 'nn'

local Model = {}

function Model:createAutoencoder(X)
  local featureSize = X:size(2)

  -- Create encoder
  self.encoder = nn.Sequential()
  self.encoder:add(nn.Linear(featureSize, 2000))
  self.encoder:add(nn.ReLU(true))
  self.encoder:add(nn.Linear(2000, 1000))
  self.encoder:add(nn.ReLU(true))
  self.encoder:add(nn.Linear(1000, 2000))
  self.encoder:add(nn.ReLU(true))
  self.encoder:add(nn.Linear(2000, 20))
  -- self.encoder:add(nn.Sigmoid(true))

  -- Create decoder
  self.decoder = nn.Sequential()
  self.decoder:add(nn.Linear(20, 2000))
  self.decoder:add(nn.ReLU(true))
  self.decoder:add(nn.Linear(2000, 1000))
  self.decoder:add(nn.ReLU(true))
  self.decoder:add(nn.Linear(1000, 2000))
  self.decoder:add(nn.ReLU(true))
  self.decoder:add(nn.Linear(2000, featureSize))
  self.decoder:add(nn.Sigmoid(true))


  -- Create autoencoder
  self.autoencoder = nn.Sequential()
  self.noiser = nn.WhiteNoise(0, 0.1) -- Add white noise to inputs during training
  self.autoencoder:add(self.noiser)
  self.autoencoder:add(self.encoder)
  self.autoencoder:add(self.decoder)

  for i=1,self.encoder:size() do
      if self.encoder.modules[i].weight then
          self.encoder.modules[i].weight = torch.randn(self.encoder.modules[i].weight:size()) * 0.01
      end
  end

  for i=1,self.decoder:size() do
      if self.decoder.modules[i].weight then
          self.decoder.modules[i].weight = torch.randn(self.decoder.modules[i].weight:size()) * 0.01
      end
  end
end

return Model
