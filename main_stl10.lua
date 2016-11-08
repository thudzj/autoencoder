  -- Load dependencies
local mnist = require 'mnist'
local optim = require 'optim'
local gnuplot = require 'gnuplot'
local image = require 'image'
local cuda = pcall(require, 'cutorch') -- Use CUDA if available
local hasCudnn, cudnn = pcall(require, 'cudnn') -- Use cuDNN if available
local npy4th = require 'npy4th'
require 'image'

require 'dpnn'
-- require 'tools_ddcrp'
require 'tools'

-- Set up Torch
print('Setting up')
torch.setdefaulttensortype('torch.FloatTensor')
torch.manualSeed(torch.random())
if cuda then
  require 'cunn'
  cutorch.manualSeed(torch.random())
  print(string.format("GPU number: %d", cutorch.getDeviceCount()))
  cutorch.setDevice(1)
  print(string.format("Using GPU %d", cutorch.getDevice()))
end

-- Choose model to train
local cmd = torch.CmdLine()
cmd:option('-model', 'DeepAE', 'Model: AE|SparseAE|DeepAE|ConvAE|UpconvAE|DenoisingAE|Seq2SeqAE|VAE|AAE')
cmd:option('-learningRate', 0.001, 'Learning rate')
cmd:option('-optimiser', 'adam', 'Optimiser')
cmd:option('-epochs', 300, 'Training epochs')
cmd:option('-sample_times', 10, 'sampling times before calculate gradient')
cmd:option('-K', 3, 'calculate gradient for K times and then averaging')
cmd:option('-n_points', 13000, 'randomly sample n_points instances from dataset to train')
cmd:option('-delta', 0.01, 'balance the loss of ae and ddcrp')
cmd:option('-alpha', 1, 'alpha')
cmd:option('-lambda', 1, 'lambda')
cmd:option('-batchSize', 250, 'batchSize')
cmd:option('-name', 'init_stl10', 'name of the trained model')
cmd:option('-task', 'ae', 'ddcrp or ae')
cmd:option('-dataset', 'mnist', 'dataset')
cmd:option('-criterion', 'mse', 'criterion')
cmd:option('-scale', 1, 'scale')
cmd:option('-gamma', 1, 'gamma')
local opt = cmd:parse(arg)
if opt.task == 'ddcrp' then
  opt.n_points = 1000
  opt.batchSize = 1000
end
local n_points = opt.n_points

local map = npy4th.loadnpz(string.format('stl_features.npz'))
XTrain = map['arr_0']
yTrain = map['arr_1']
N = XTrain:size(1)
if cuda then
  XTrain = XTrain:cuda()
end

-- XTrain = XTrain:cdiv(torch.norm(XTrain, 2, 2):expandAs(XTrain))
-- XTrain = (XTrain - torch.min(XTrain)) / (torch.max(XTrain) - torch.min(XTrain))   -- 

XTrain = XTrain[{{1, n_points}}]
yTrain = yTrain[{{1, n_points}}]

torch.save('data/image.t7', XTrain:double())
torch.save('data/label.t7', yTrain)

local label_unique = {}
for i = 1, n_points do
  if label_unique[yTrain[i]+1] == nil then
    label_unique[yTrain[i]+1] = 1
  else
    label_unique[yTrain[i]+1] = label_unique[yTrain[i]+1] + 1
  end
end
print(label_unique)

indices = (yTrain + 1):view(-1, 1):long()
one_hot = torch.zeros(n_points, #label_unique)
one_hot = one_hot:scatter(2, indices, 1):cuda()

-- Create model
local model = nil
if not path.exists(string.format('trained/%s', opt.name)) then
  Model = require ('models/' .. opt.model)
  Model:createAutoencoder(XTrain, XTrain:size(2))
else
  print('Loading existed models ' .. string.format('trained/%s', opt.name))
  Model = torch.load(string.format('trained/%s', opt.name))
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

if opt.task == 'ddcrp' then
  encoder:forward(XTrain)
  torch.save('data/feature.t7', encoder.output:double())
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

-- local kappa_0 = 1
-- local nu_0 = 20 -- not sure
-- local mu_0 = torch.CudaTensor(opt.feature_size):fill(0)
-- local Lambda_0 = torch.eye(opt.feature_size):cuda()
-- local Lambda_0_det_pow_nu0_div_2 = 1

-- Get parameters
local theta, gradTheta = autoencoder:getParameters()

-- Create loss
local criterion = nn.BCECriterion() --nn.MSECriterion()-- nn.MSECriterion()--
if opt.criterion == 'mse' then
  criterion = nn.MSECriterion()
end

if cuda then
  criterion:cuda()
end

-- Create optimiser function evaluation
local x -- Minibatch
local y
local epoch_=0
-- local tmp_label
local feval = function(params)
  if theta ~= params then
    theta:copy(params)
  end
  -- Zero gradients
  gradTheta:zero()

  -- Reconstruction phase
  -- Forward propagation
  local xHat = autoencoder:forward(x) -- Reconstruction
  local loss = criterion:forward(xHat, y)

  -- local maxs, indices = torch.max(xHat, 2)
  -- indices = indices:long():view(-1)
  -- print(indices:eq((tmp_label+1):long()):sum()/x:size(1))
  -- Backpropagation
  local gradLoss = criterion:backward(xHat, y)
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
  local loss = criterion:forward(xHat, y)
  print(string.format("   [%s], ae-loss: %f", os.date("%c", os.time()), loss))
  -- Backpropagation
  local gradLoss = criterion:backward(xHat, y)
  autoencoder:backward(x, gradLoss)

  --encoder:evaluate()
  local z = encoder:forward(x)
  local scale = 1 --(torch.max(z) - torch.min(z)) * opt.scale
  z = z:div(scale)
  local z_d = z:double()
  -- print(z[{{1,5}}])
  -- print(z:size())
  
  --encoder:training()
  for ite = 1, opt.sample_times do
      --n_tables = gibbs_sample(z_d, belong, connect, tables, connected, n_tables, opt.alpha,kappa_0, nu_0, mu_0, Lambda_0, Lambda_0_det_pow_nu0_div_2)
      n_tables = gibbs_sample(z_d, belong, connect, tables, connected, n_tables, opt.alpha,opt.lambda)
  end

  for ite = 1, opt.K do
      --n_tables = gibbs_sample(z_d, belong, connect, tables, connected, n_tables, opt.alpha, kappa_0, nu_0, mu_0, Lambda_0, Lambda_0_det_pow_nu0_div_2)
      n_tables = gibbs_sample(z_d, belong, connect, tables, connected, n_tables, opt.alpha,opt.lambda)
      --l1_loss, l2_loss, dz = cal_gradient(z, connect, tables, opt.alpha, kappa_0, nu_0, mu_0, Lambda_0, Lambda_0_det_pow_nu0_div_2)
      local l1_loss, l2_loss, S_loss, dz, S_dz = cal_gradient(z, connect, tables, opt.alpha, opt.lambda, epoch_, ite)
      dz = dz * opt.delta / opt.K / n_points
      S_dz = S_dz * opt.gamma / opt.K
      loss = loss + (l1_loss + l2_loss) * opt.delta / opt.K / n_points + S_loss * opt.gamma / opt.K
      print(
        string.format("   [%s], loss1: %f, loss2: %f, S_loss: %f, loss: %f", os.date("%c", os.time()), 
          l1_loss * opt.delta / opt.K / n_points, 
          l2_loss * opt.delta / opt.K / n_points, 
          S_loss * opt.gamma / opt.K, 
          (l1_loss + l2_loss) * opt.delta / opt.K / n_points + S_loss * opt.gamma / opt.K
        )
      )
      print(dz[{{1,10}, {1, 10}}])
      encoder:backward(x, (dz + S_dz) * scale)
  end

  return loss, gradTheta
end

-- Train
print('Training')
autoencoder:training()
local optimParams = {learningRate = opt.learningRate}
local __, loss
local losses, advLosses = {}, {}

for epoch = 1, opt.epochs do
  epoch_ = epoch
  print('Epoch ' .. epoch .. '/' .. opt.epochs)
  for n = 1, n_points, opt.batchSize do
    -- Get minibatch
    x = XTrain:narrow(1, n, opt.batchSize)
    y = x --one_hot:narrow(1, n, opt.batchSize)
    -- tmp_label = yTrain:narrow(1, n, opt.batchSize)

    -- Optimise
    if opt.task == 'ddcrp' then
      __, loss = optim[opt.optimiser](ddcrpFeval, theta, optimParams)
      losses[#losses + 1] = loss[1]
    --   optimParams.learningRate = 0.0001
    --   for ite = 1, 1 do
    --     __, loss = optim[opt.optimiser](feval2, theta, optimParams)
    --   end

    --   optimParams.learningRate = opt.learningRate
    --   for ite = 1, 10 do
    --     __, loss = optim[opt.optimiser](feval, theta, optimParams)
    --   end
    --   print(string.format("  loss: %4f", loss[1]))
    --   losses[#losses + 1] = loss[1]
    else
      __, loss = optim[opt.optimiser](feval, theta, optimParams)
      losses[#losses + 1] = loss[1]
    end
  end
  if opt.task == 'ddcrp' then
    evaluate_cluster(tables, yTrain, epoch, loss, n_points, #label_unique)
  else
    print(string.format("  loss: %4f", loss[1]))
  end

  -- Plot training curve(s)
  local plots = {{'Autoencoder', torch.linspace(1, #losses, #losses), torch.Tensor(losses), '-'}}
  gnuplot.pngfigure('cifar.png')
  gnuplot.plot(table.unpack(plots))
  gnuplot.ylabel('Loss')
  gnuplot.xlabel('Batch #')
  gnuplot.plotflush()

  if epoch % 30 == 0 and opt.task == 'ddcrp' then
    torch.save('trained/' .. opt.name .. string.format('_%d_%2f_%2f_ddcrp', epoch, opt.alpha, opt.lambda), Model)
    torch.save(string.format("results/tables_at_epoch_%d", epoch), tables)
  end
end


-- Test
-- print('Testing')
-- x = XTrain:narrow(1, 1, 1000)
-- y = yTrain:narrow(1, 1, 1000)
-- autoencoder:evaluate()
-- local xHat = autoencoder:forward(x)
-- maxs, indices = torch.max(xHat, 2)

-- indices = indices:long():view(-1)
-- print(indices:eq((y+1):long()):sum()/1000)



if opt.task == 'ae' then
  torch.save(string.format('trained/%s', opt.name), Model)
end

if opt.ddcrp == 'ddcrp' then
  encoder:forward(XTrain)
  torch.save(string.format('data/feature_%2f_%2f.t7', alpha, lambda), encoder.output:double())
end
