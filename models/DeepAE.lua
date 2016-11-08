local nn = require 'nn'

local Model = {}

function Model:createAutoencoder(X, featuresz)
  local featureSize = featuresz or X:size(2) * X:size(3)

  -- Create encoder
  self.encoder = nn.Sequential()
  self.encoder:add(nn.View(-1, featureSize))
  self.encoder:add(nn.Linear(featureSize, 500))
  self.encoder:add(nn.ReLU(true))
  self.encoder:add(nn.Linear(500, 500))
  self.encoder:add(nn.ReLU(true))
  self.encoder:add(nn.Linear(500, 2000))
  self.encoder:add(nn.ReLU(true))
  self.encoder:add(nn.Linear(2000, 10))
  -- self.encoder:add(nn.Sigmoid(true))

  -- Create decoder
  self.decoder = nn.Sequential()
  self.decoder:add(nn.Linear(10, 2000))
  self.decoder:add(nn.ReLU(true))
  self.decoder:add(nn.Linear(2000, 500))
  self.decoder:add(nn.ReLU(true))
  self.decoder:add(nn.Linear(500, 500))
  self.decoder:add(nn.ReLU(true))
  self.decoder:add(nn.Linear(500, featureSize))



  -- self.decoder:add(nn.Sigmoid(true))
  if featuresz == nil then
    self.decoder:add(nn.View(X:size(2), X:size(3)))
  end

  -- Create autoencoder
  self.autoencoder = nn.Sequential()
  -- self.noiser = nn.WhiteNoise(0, 0.2) -- Add white noise to inputs during training
  -- self.autoencoder:add(self.noiser)
  self.autoencoder:add(self.encoder)
  self.autoencoder:add(self.decoder)

  -- for i=1,self.encoder:size() do
  --     if self.encoder.modules[i].weight then
  --         self.encoder.modules[i].weight = torch.randn(self.encoder.modules[i].weight:size()) * 0.01
  --     end
  -- end

  -- for i=1,self.decoder:size() do
  --     if self.decoder.modules[i].weight then
  --         self.decoder.modules[i].weight = torch.randn(self.decoder.modules[i].weight:size()) * 0.01
  --     end
  -- end
end

return Model
