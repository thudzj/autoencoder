  -- Load dependencies
local mnist = require 'mnist'
local optim = require 'optim'
local gnuplot = require 'gnuplot'
local image = require 'image'
local cuda = pcall(require, 'cutorch') -- Use CUDA if available
local hasCudnn, cudnn = pcall(require, 'cudnn') -- Use cuDNN if available
require 'dpnn'
require 'tools'

-- Set up Torch
print('Setting up')
torch.setdefaulttensortype('torch.FloatTensor')
torch.manualSeed(1)
if cuda then
  require 'cunn'
  cutorch.manualSeed(torch.random())
  print(string.format("GPU number: %d", cutorch.getDeviceCount()))
  cutorch.setDevice(1)
  print(string.format("Using GPU %d", cutorch.getDevice()))
end

-- Load MNIST data
local XTrain = mnist.traindataset().data:float():div(255) -- Normalise to [0, 1]
local yTrain = mnist.traindataset().label
local XTest = mnist.testdataset().data:float():div(255)
local N = XTrain:size(1)
if cuda then
  XTrain = XTrain:cuda()
  XTest = XTest:cuda()
end

-- Choose model to train
local cmd = torch.CmdLine()
cmd:option('-model', 'AE', 'Model: AE|SparseAE|DeepAE|ConvAE|UpconvAE|DenoisingAE|Seq2SeqAE|VAE|AAE')
cmd:option('-learningRate', 0.001, 'Learning rate')
cmd:option('-optimiser', 'adam', 'Optimiser')
cmd:option('-epochs', 300, 'Training epochs')
cmd:option('-sample_times', 7, 'sampling times before calculate gradient')
cmd:option('-K', 3, 'calculate gradient for K times and then averaging')
cmd:option('-n_points', 1000, 'randomly sample n_points instances from dataset to train')
cmd:option('-delta', 0.003, 'balance the loss of ae and ddcrp')
cmd:option('-alpha', 1, 'alpha')
cmd:option('-lambda', 1, 'lambda')
cmd:option('-batchSize', 150, 'batchSize')
cmd:option('-name', 'init', 'name of the trained model')
cmd:option('-task', 'ddcrp', 'ddcrp or ae')
local opt = cmd:parse(arg)
local n_points = opt.n_points

local index = torch.randperm(N)[{{1, n_points}}]:long()
XTrain = XTrain:index(1, index):cuda()
yTrain = yTrain:index(1, index)

print(XTrain:size())

-- Create model
local model = nil
if not path.exists('trained/' .. opt.name) then
  Model = require ('models/' .. opt.model)
  Model:createAutoencoder(XTrain)
else
  Model = torch.load('trained/' .. opt.name)
end

local autoencoder = Model.autoencoder
local encoder = Model.encoder
local decoder = Model.decoder
if cuda then
  autoencoder:cuda()
  encoder:cuda()
  decoder:cuda()
  -- Use cuDNN if available
  if hasCudnn then
    cudnn.convert(autoencoder, cudnn)
    cudnn.convert(encoder, cudnn)
    cudnn.convert(decoder, cudnn)
  end
end

-- Create tables and the relations between points
local belong = torch.range(1, n_points):int()
local connect = torch.range(1, n_points):int()
local n_tables = n_points
local tables = {}
for i = 1, n_points do
    tables[i] = {}
    table.insert(tables[i], i)
end
local connected = {}
for i = 1, n_points do
    connected[i] = {}
    table.insert(connected[i], i)
end

-- Create adversary (if needed)
local adversary
if opt.model == 'AAE' then
  Model:createAdversary()
  adversary = Model.adversary
  if cuda then
    adversary:cuda()
    -- Use cuDNN if available
    if hasCudnn then
      cudnn.convert(adversary, cudnn)
    end
  end
end

-- Get parameters
local theta, gradTheta = autoencoder:getParameters()
local thetaAdv, gradThetaAdv
if opt.model == 'AAE' then
  thetaAdv, gradThetaAdv = adversary:getParameters()
end

-- Create loss
local criterion = nn.BCECriterion()
if cuda then
  criterion:cuda()
end

-- Create optimiser function evaluation
local x -- Minibatch
local feval = function(params)
  if theta ~= params then
    theta:copy(params)
  end
  -- Zero gradients
  gradTheta:zero()
  if opt.model == 'AAE' then
    gradThetaAdv:zero()
  end

  -- Reconstruction phase
  -- Forward propagation
  local xHat = autoencoder:forward(x) -- Reconstruction
  local loss = criterion:forward(xHat, x)
  -- Backpropagation
  local gradLoss = criterion:backward(xHat, x)
  autoencoder:backward(x, gradLoss)

  -- Regularization phase
  if opt.model == 'Seq2SeqAE' then
    -- Clamp RNN gradients to prevent exploding gradients
    gradTheta:clamp(-10, 10)
  elseif opt.model == 'VAE' then
    local encoder = Model.encoder

    -- Optimize Gaussian KL-Divergence between inference model and prior: DKL(q(z)||N(0, I)) = log(σ2/σ1) + ((σ1^2 - σ2^2) + (μ1 - μ2)^2) / 2σ2^2
    local nElements = xHat:nElement()
    local mean, logVar = table.unpack(encoder.output)
    local var = torch.exp(logVar)
    local KLLoss = 0.5 * torch.sum(torch.pow(mean, 2) + var - logVar - 1)
    KLLoss = KLLoss / nElements -- Normalise loss (same normalisation as BCECriterion)
    loss = loss + KLLoss
    local gradKLLoss = {mean / nElements, 0.5*(var - 1) / nElements}  -- Normalise gradient of loss (same normalisation as BCECriterion)
    encoder:backward(x, gradKLLoss)
  elseif opt.model == 'AAE' then
    local encoder = Model.encoder
    local real = torch.Tensor(opt.batchSize, Model.zSize):normal(0, 1):typeAs(XTrain) -- Real samples ~ N(0, 1)
    local YReal = torch.ones(opt.batchSize):typeAs(XTrain) -- Labels for real samples
    local YFake = torch.zeros(opt.batchSize):typeAs(XTrain) -- Labels for generated samples

    -- Train adversary to maximise log probability of real samples: max_D log(D(x))
    local pred = adversary:forward(real)
    local realLoss = criterion:forward(pred, YReal)
    local gradRealLoss = criterion:backward(pred, YReal)
    adversary:backward(real, gradRealLoss)

    -- Train adversary to minimise log probability of fake samples: max_D log(1 - D(G(x)))
    pred = adversary:forward(encoder.output)
    local fakeLoss = criterion:forward(pred, YFake)
    advLoss = realLoss + fakeLoss
    local gradFakeLoss = criterion:backward(pred, YFake)
    local gradFake = adversary:backward(encoder.output, gradFakeLoss)

    -- Train encoder (generator) to play a minimax game with the adversary (discriminator): min_G max_D log(1 - D(G(x)))
    local minimaxLoss = criterion:forward(pred, YReal)
    loss = loss + minimaxLoss
    local gradMinimaxLoss = criterion:backward(pred, YReal)
    local gradMinimax = adversary:updateGradInput(encoder.output, gradMinimaxLoss) -- Do not calculate gradient wrt adversary parameters
    encoder:backward(x, gradMinimax)
  end

  return loss, gradTheta
end

local ddcrpFeval = function(params)
  if theta ~= params then
    theta:copy(params)
  end
  -- Zero gradients
  gradTheta:zero()

  local xHat = autoencoder:forward(x) -- Reconstruction
  local loss = criterion:forward(xHat, x)
  print(string.format("   [%s], ae-loss: %f", os.date("%c", os.time()), loss))
  -- Backpropagation
  local gradLoss = criterion:backward(xHat, x)
  autoencoder:backward(x, gradLoss)

  --encoder:evaluate()
  local z = encoder:forward(x)
  local z_d = z:double()
  -- print(z[{{1,5}}])
  -- print(z:size())
  
  --encoder:training()
  for ite = 1, opt.sample_times do
      n_tables = gibbs_sample(z_d, belong, connect, tables, connected, n_tables, opt.alpha, opt.lambda)
  end

  for ite = 1, opt.K do
      n_tables = gibbs_sample(z_d, belong, connect, tables, connected, n_tables, opt.alpha, opt.lambda)
      l1_loss, l2_loss, dz = cal_gradient(z, connect, tables, opt.alpha, opt.lambda)
      dz = dz * opt.delta / opt.K / n_points
      loss = loss + (l1_loss + l2_loss) * opt.delta / opt.K / n_points
      print(string.format("   [%s], loss1: %f, loss2: %f", os.date("%c", os.time()), l1_loss * opt.delta / opt.K / n_points, l2_loss * opt.delta / opt.K / n_points))

      encoder:backward(x, dz)
  end

  return loss, gradTheta
end

local advFeval = function(params)
  if thetaAdv ~= params then
    thetaAdv:copy(params)
  end

  return advLoss, gradThetaAdv
end

-- Train
print('Training')
autoencoder:training()
local optimParams = {learningRate = opt.learningRate}
local advOptimParams = {learningRate = opt.learningRate}
local __, loss
local losses, advLosses = {}, {}

for epoch = 1, opt.epochs do
  print('Epoch ' .. epoch .. '/' .. opt.epochs)
  for n = 1, n_points, opt.batchSize do
    -- Get minibatch
    x = XTrain:narrow(1, n, opt.batchSize)

    -- Optimise
    if opt.task == 'ddcrp' then
      __, loss = optim[opt.optimiser](ddcrpFeval, theta, optimParams)
      losses[#losses + 1] = loss[1]
    else
      __, loss = optim[opt.optimiser](feval, theta, optimParams)
      losses[#losses + 1] = loss[1]
    end

    -- Train adversary
    if opt.model == 'AAE' then
      __, loss = optim[opt.optimiser](advFeval, thetaAdv, advOptimParams)     
      advLosses[#advLosses + 1] = loss[1]
    end
  end
  if opt.task == 'ddcrp' then
    evaluate_cluster(tables, yTrain, epoch, loss, n_points)
  else
    print(string.format("  loss: %4f", loss[1]))
  end

  -- Plot training curve(s)
  local plots = {{'Autoencoder', torch.linspace(1, #losses, #losses), torch.Tensor(losses), '-'}}
  gnuplot.pngfigure('Training.png')
  gnuplot.plot(table.unpack(plots))
  gnuplot.ylabel('Loss')
  gnuplot.xlabel('Batch #')
  gnuplot.plotflush()

  if epoch % 30 == 0 then
    torch.save('trained/' .. opt.name .. string.format('_%d_%f_ddcrp', epoch, opt.alpha), Model)
    torch.save(string.format("results/tables_at_epoch_%d", epoch), tables)
  end
end


-- Test
print('Testing')
x = XTest:narrow(1, 1, 10)
local xHat
if opt.model == 'DenoisingAE' or opt.model == 'DeepAE' then
  -- Normally this should be switched to evaluation mode, but this lets us extract the noised version
  autoencoder:evaluate()
  xHat = decoder:forward(encoder:forward(x))

  -- Extract noised version from denoising AE
  -- x = Model.noiser.output
else
  autoencoder:evaluate()
  xHat = autoencoder:forward(x)
end

-- Plot reconstructions
image.save('Reconstructions.png', torch.cat(image.toDisplayTensor(x, 2, 10), image.toDisplayTensor(xHat, 2, 10), 1))
if opt.task ~= 'ddcrp' then
  torch.save('trained/' .. opt.name, Model)
end

-- Plot samples
if opt.model == 'VAE' or opt.model == 'AAE' then
  local decoder = Model.decoder
  local height, width = XTest:size(2), XTest:size(3)
  local samples = torch.Tensor(15 * height, 15 * width):typeAs(XTest)
  local std = 0.05

  -- Sample 15 x 15 points
  for i = 1, 15  do
    for j = 1, 15 do
      local sample = torch.Tensor({2 * i * std - 16 * std, 2 * j * std - 16 * std}):typeAs(XTest):view(1, 2) -- Minibatch of 1 for batch normalisation
      samples[{{(i-1) * height + 1, i * height}, {(j-1) * width + 1, j * width}}] = decoder:forward(sample)
    end
  end
  image.save('Samples.png', samples)
end
