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
cmd:option('-model', 'ConvAE', 'Model: AE|SparseAE|DeepAE|ConvAE|UpconvAE|DenoisingAE|Seq2SeqAE|VAE|AAE')
cmd:option('-learningRate', 0.001, 'Learning rate')
cmd:option('-optimiser', 'adam', 'Optimiser')
cmd:option('-epochs', 300, 'Training epochs')
cmd:option('-sample_times', 7, 'sampling times before calculate gradient')
cmd:option('-K', 3, 'calculate gradient for K times and then averaging')
cmd:option('-n_points', 50000, 'randomly sample n_points instances from dataset to train')
cmd:option('-delta', 0.003, 'balance the loss of ae and ddcrp')
cmd:option('-alpha', 1, 'alpha')
cmd:option('-lambda', 1, 'lambda')
cmd:option('-batchSize', 250, 'batchSize')
cmd:option('-name', 'init', 'name of the trained model')
cmd:option('-task', 'ddcrp', 'ddcrp or ae')
cmd:option('-dataset', 'mnist', 'dataset')
cmd:option('-feature_size', 10, '# features')
cmd:option('-criterion', 'bce', 'criterion')
local opt = cmd:parse(arg)
local n_points = opt.n_points

-- download dataset
if not paths.dirp('cifar-10-batches-t7') then
   local www = 'http://torch7.s3-website-us-east-1.amazonaws.com/data/cifar-10-torch.tar.gz'
   local tar = paths.basename(www)
   os.execute('wget ' .. www .. '; '.. 'tar xvf ' .. tar)
end

local N = 60000
-- load dataset
trainData = {
   data = torch.Tensor(N, 3072),
   labels = torch.Tensor(N),
   size = function() return N end
}
for i = 0,4 do
   subset = torch.load('cifar-10-batches-t7/data_batch_' .. (i+1) .. '.t7', 'ascii')
   trainData.data[{ {i*10000+1, (i+1)*10000} }] = subset.data:t()
   trainData.labels[{ {i*10000+1, (i+1)*10000} }] = subset.labels
end

subset = torch.load('cifar-10-batches-t7/test_batch.t7', 'ascii')
trainData.data[{ {5*10000+1, (5+1)*10000} }] = subset.data:t()
trainData.labels[{ {5*10000+1, (5+1)*10000} }] = subset.labels
-- trainData.labels = trainData.labels

-- trainData.data = trainData.data:reshape(N,3,32,32)
-- local XTrain = torch.Tensor(N, 1024)
-- for i = 1, N do
--   XTrain[i] = image.rgb2y(trainData.data[i]):reshape(1024):div(255)
-- end
local XTrain = trainData.data[{{1, N}}]
local XTrain_grey = torch.Tensor(N, 1024)
for i = 1, N do
  XTrain_grey[i] = image.rgb2y(XTrain[i]:reshape(3, 32, 32)):reshape(1024):div(255)
end
XTrain = XTrain:div(255)
local yTrain = trainData.labels[{ {1,N} }]

-- preprocess trainSet
-- trainData.data = trainData.data:reshape(N,3,32,32)
-- normalization = nn.SpatialContrastiveNormalization(1, image.gaussian1D(7))
-- for i = 1,trainData:size() do
--    -- rgb -> yuv
--    local rgb = trainData.data[i]
--    local yuv = image.rgb2yuv(rgb)
--    -- normalize y locally:
--    yuv[1] = normalization(yuv[{{1}}])
--    trainData.data[i] = yuv
-- end
-- -- normalize u globally:
-- mean_u = trainData.data[{ {},2,{},{} }]:mean()
-- std_u = trainData.data[{ {},2,{},{} }]:std()
-- trainData.data[{ {},2,{},{} }]:add(-mean_u)
-- trainData.data[{ {},2,{},{} }]:div(-std_u)
-- -- normalize v globally:
-- mean_v = trainData.data[{ {},3,{},{} }]:mean()
-- std_v = trainData.data[{ {},3,{},{} }]:std()
-- trainData.data[{ {},3,{},{} }]:add(-mean_v)
-- trainData.data[{ {},3,{},{} }]:div(-std_v)

if cuda then
    XTrain = XTrain:cuda()
    XTrain_grey = XTrain_grey:cuda()
end

local index = torch.randperm(N)[{{1, n_points}}]:long()
XTrain = XTrain:index(1, index)
yTrain = yTrain:index(1, index)
XTrain_grey = XTrain_grey:index(1, index)

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

-- Create model
local model = nil
if not path.exists(string.format('trained/%s_%d', opt.name, opt.feature_size)) then
  Model = require ('models/' .. opt.model)
  Model:createAutoencoder(XTrain, opt.feature_size)
else
  print('Loading existed models ' .. string.format('trained/%s_%d', opt.name, opt.feature_size))
  Model = torch.load(string.format('trained/%s_%d', opt.name, opt.feature_size))
end

local autoencoder = Model.autoencoder
local encoder = Model.encoder
local decoder = Model.decoder

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

local kappa_0 = 1
local nu_0 = 20 -- not sure
local mu_0 = torch.CudaTensor(opt.feature_size):fill(0)
local Lambda_0 = torch.eye(opt.feature_size):cuda()
local Lambda_0_det_pow_nu0_div_2 = 1

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
local x_grey
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
  local loss = criterion:forward(xHat, x_grey)
  -- Backpropagation
  local gradLoss = criterion:backward(xHat, x_grey)
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
  local loss = criterion:forward(xHat, x_grey)
  print(string.format("   [%s], ae-loss: %f", os.date("%c", os.time()), loss))
  -- Backpropagation
  local gradLoss = criterion:backward(xHat, x_grey)
  --autoencoder:backward(x, gradLoss)

  --encoder:evaluate()
  local z = encoder:forward(x)
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
      l1_loss, l2_loss, dz = cal_gradient(z, connect, tables, opt.alpha, opt.lambda)
      dz = dz * opt.delta / opt.K / n_points
      loss = loss + (l1_loss + l2_loss) * opt.delta / opt.K / n_points
      print(string.format("   [%s], loss1: %f, loss2: %f", os.date("%c", os.time()), l1_loss * opt.delta / opt.K / n_points, l2_loss * opt.delta / opt.K / n_points))

      -- print(dz[{{1,10}, {1, 10}}])
      encoder:backward(x, dz)
  end

  return loss, gradTheta
end

local feval2 = function(params)
  if theta ~= params then
    theta:copy(params)
  end
  -- Zero gradients
  gradTheta:zero()

  local xHat = autoencoder:forward(x) -- Reconstruction
  local loss = criterion:forward(xHat, x_grey)
  print(string.format("   [%s], ae-loss: %f", os.date("%c", os.time()), loss))

  --encoder:evaluate()
  local z = encoder:forward(x)
  local z_d = z:double()

  for ite = 1, opt.sample_times do
      --n_tables = gibbs_sample(z_d, belong, connect, tables, connected, n_tables, opt.alpha,kappa_0, nu_0, mu_0, Lambda_0, Lambda_0_det_pow_nu0_div_2)
      n_tables = gibbs_sample(z_d, belong, connect, tables, connected, n_tables, opt.alpha,opt.lambda)
  end

  for ite = 1, opt.K do
      --n_tables = gibbs_sample(z_d, belong, connect, tables, connected, n_tables, opt.alpha, kappa_0, nu_0, mu_0, Lambda_0, Lambda_0_det_pow_nu0_div_2)
      n_tables = gibbs_sample(z_d, belong, connect, tables, connected, n_tables, opt.alpha,opt.lambda)
      --l1_loss, l2_loss, dz = cal_gradient(z, connect, tables, opt.alpha, kappa_0, nu_0, mu_0, Lambda_0, Lambda_0_det_pow_nu0_div_2)
      l1_loss, l2_loss, dz = cal_gradient(z, connect, tables, opt.alpha, opt.lambda)
      dz = dz * opt.delta / opt.K / n_points
      loss = loss + (l1_loss + l2_loss) * opt.delta / opt.K / n_points
      print(string.format("   [%s], loss1: %f, loss2: %f, loss: %f", os.date("%c", os.time()), l1_loss * opt.delta / opt.K / n_points, l2_loss * opt.delta / opt.K / n_points, (l1_loss + l2_loss) * opt.delta / opt.K / n_points))

      encoder:backward(x, dz)
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
  print('Epoch ' .. epoch .. '/' .. opt.epochs)
  for n = 1, n_points, opt.batchSize do
    -- Get minibatch
    x = XTrain:narrow(1, n, opt.batchSize)
    x_grey= XTrain_grey:narrow(1, n, opt.batchSize)

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
    -- else
    --   __, loss = optim[opt.optimiser](feval, theta, optimParams)
    --   losses[#losses + 1] = loss[1]
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
print('Testing')
x = XTrain:narrow(1, 1, 10)
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

x = x:reshape(10, 3, 32, 32)
xHat = xHat:reshape(10, 1, 32, 32)

-- Plot reconstructions
image.save(string.format('sample_%d.png', opt.feature_size), image.toDisplayTensor(x, 2, 10))
image.save(string.format('sample_recon_%d.png', opt.feature_size), image.toDisplayTensor(xHat, 2, 10))


if opt.task == 'ae' then
  torch.save(string.format('trained/%s_%d', opt.name, opt.feature_size), Model)
end

if opt.ddcrp == 'ddcrp' then
  encoder:forward(XTrain)
  torch.save(string.format('data/feature_%2f_%2f.t7', alpha, lambda), encoder.output:double())
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
