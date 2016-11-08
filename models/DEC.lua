local nn = require 'nn'
require 'loadcaffe'
local Model = {}

function Model:createAutoencoder(X)

  local featureSize = X:size(2) * X:size(3)

  -- Create encoder
  self.encoder = nn.Sequential()
  self.encoder:add(nn.View(-1, featureSize))
  --self.encoder:add(nn.Dropout(0.2))
  self.encoder:add(nn.Linear(featureSize, 500))
  self.encoder:add(nn.ReLU(true))
  --self.encoder:add(nn.Dropout(0.2))
  self.encoder:add(nn.Linear(500, 500))
  self.encoder:add(nn.ReLU(true))

  ----self.encoder:add(nn.Dropout(0.2))
  self.encoder:add(nn.Linear(500, 2000))
  self.encoder:add(nn.ReLU(true))

  ----self.encoder:add(nn.Dropout(0.2))
  self.encoder:add(nn.Linear(2000, 10))
  ---- self.encoder:add(nn.Sigmoid(true))

  ---- Create decoder
  self.decoder = nn.Sequential()
  ----self.decoder:add(nn.Dropout(0.2))
  self.decoder:add(nn.Linear(10, 2000))
  self.decoder:add(nn.ReLU(true))

  ----self.decoder:add(nn.Dropout(0.2))
  self.decoder:add(nn.Linear(2000, 500))
  self.decoder:add(nn.ReLU(true))

  ----self.decoder:add(nn.Dropout(0.2))
  self.decoder:add(nn.Linear(500, 500))
  self.decoder:add(nn.ReLU(true))

  --self.decoder:add(nn.Dropout(0.2))
  self.decoder:add(nn.Linear(500, featureSize))
  --self.decoder:add(nn.ReLU(true))
  --self.decoder:add(nn.Sigmoid(true))
  self.decoder:add(nn.View(X:size(2), X:size(3)))

  -- Create autoencoder
  self.autoencoder = nn.Sequential()
  --self.noiser = nn.WhiteNoise(0, 0.5) -- Add white noise to inputs during training
  --self.autoencoder:add(self.noiser)
  self.autoencoder:add(self.encoder)
  self.autoencoder:add(self.decoder)
  
  self:init()
end

function Model:init()

  local model = loadcaffe.load('models/net.prototxt', 'models/save_iter_100000.caffemodel', 'ccn2')
  print(model)
  targets = self.autoencoder:findModules('nn.Linear')
  sources = model:findModules('nn.Linear')
  for i =1, #targets do
    targets[i].weight:copy(sources[i].weight)
    targets[i].bias:copy(sources[i].bias)
  end

  --for k,v in pairs(self.encoder:findModules('nn.Linear')) do
  --  v.weight:normal(0, 0.01)
  --  v.bias:zero()
  --end
  --for k,v in pairs(self.decoder:findModules('nn.Linear')) do
  --  v.weight:normal(0, 0.01)
  --  v.bias:zero()
  --end
end


return Model
