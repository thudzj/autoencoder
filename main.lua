  -- Load dependencies
local mnist = require 'mnist'
local optim = require 'optim'
local gnuplot = require 'gnuplot'
local image = require 'image'
local cuda = pcall(require, 'cutorch') -- Use CUDA if available
local hasCudnn, cudnn = pcall(require, 'cudnn') -- Use cuDNN if available
local npy4th = require 'npy4th'
local nn = require 'nn'

require 'dpnn'
require 'tools'
require 'distributions'

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
cmd:option('-model', 'DEC', 'Model: AE|SparseAE|DeepAE|ConvAE|UpconvAE|DenoisingAE|Seq2SeqAE|VAE|AAE')
cmd:option('-learningRate', 0.0001, 'Learning rate')
cmd:option('-optimiser', 'adam', 'Optimiser')
cmd:option('-epochs', 300, 'Training epochs')
cmd:option('-sample_times', 12, 'sampling times before calculate gradient')
cmd:option('-K', 3, 'calculate gradient for K times and then averaging')
cmd:option('-n_points', 60000, 'randomly sample n_points instances from dataset to train')
cmd:option('-delta', 0.03, 'balance the loss of ae and ddcrp')
cmd:option('-alpha', 1, 'alpha')
cmd:option('-batchSize', 250, 'batchSize')
cmd:option('-name', 'mnist_joint', 'name of the trained model')
cmd:option('-task', 'ddcrp', 'ddcrp or ae')
cmd:option('-dataset', 'mnist', 'dataset')
cmd:option('-gamma', '0', 'gamma')
cmd:option('-criterion', 'mse','criterion')
local opt = cmd:parse(arg)

if opt.task == 'ddcrp' then
  opt.n_points = 1000
  opt.batchSize = 1000
end
local n_points = opt.n_points
local feature_size = 0

local XTrain = nil
local yTrain = nil
local XTest = nil
local N = nil
-- Load MNIST data
if opt.dataset == 'mnist' then
  XTrain = mnist.traindataset().data:float():div(50) -- Normalise to [0, 1]
  yTrain = mnist.traindataset().label
  XTest = mnist.testdataset().data:float():div(50)
  yTest = mnist.testdataset().label
  N = XTrain:size(1)
  if cuda then
    XTrain = XTrain:cuda()
    XTest = XTest:cuda()
  end
else
  local map = npy4th.loadnpz(string.format('%s.npz', opt.dataset))
  XTrain = map['arr_0']
  yTrain = map['arr_1']
  XTest = XTrain
  N = XTrain:size(1)
  if cuda then
    XTrain = XTrain:cuda()
    XTest = XTrain
  end
  print(XTrain:size())
end

local index = torch.randperm(N)[{{1, n_points}}]:long()
XTrain = XTrain:index(1, index):cuda()
yTrain = yTrain:index(1, index)

-- torch.save('data/image.t7', XTrain:double())
-- torch.save('data/label.t7', yTrain)

local label_unique = {}
for i = 1, n_points do
  if label_unique[yTrain[i]+1] == nil then
    label_unique[yTrain[i]+1] = 1
  else
    label_unique[yTrain[i]+1] = label_unique[yTrain[i]+1] + 1
  end
end
print(#label_unique)

-- Create model

Model = require ('models/' .. opt.model)
Model:createAutoencoder(XTrain)

local autoencoder = Model.autoencoder
local encoder = Model.encoder
local normalized_encoder = nn.Sequential()
normalized_encoder:add(encoder)
normalized_encoder:add(nn.Normalize(2))
if cuda then
  autoencoder:cuda()
  encoder:cuda()
  normalized_encoder:cuda()
  -- Use cuDNN if available
  if hasCudnn then
    cudnn.convert(autoencoder, cudnn)
    cudnn.convert(encoder, cudnn)
    cudnn.convert(normalized_encoder, cudnn)
  end
end


if opt.task == 'ddcrp' then
  normalized_encoder:forward(XTrain)
  feature_size = normalized_encoder.output:size(2)
  torch.save('data/feature.t7', normalized_encoder.output:double())
  torch.save('data/label.t7', yTrain)
end

-- Create tables and the relations between points
local belong = torch.range(1, n_points):int()
local tables = {}

tables[1] = {}
for i = 1, n_points do
  table.insert(tables[1], i)
  belong[i] = 1
end
local means = {}
local covs = {}

-- local n_tables = n_points
-- belong = torch.range(1, n_points):int()
-- for i = 1, n_points do
--     tables[i] = {}
--     table.insert(tables[i], i)
-- end
-- local connect = torch.range(1, n_points):int()
-- local connected = {}
-- for i = 1, n_points do
--     connected[i] = {}
--     table.insert(connected[i], i)
-- end

local mu_0 = torch.CudaTensor({-0.158365442515, 0.0377739193953, -0.0147440236072, 0.074388357756, -0.153201982234, -0.0842131485674, 0.201350941488, 0.0575697282466, 0.18518343795, -0.0174676928583, }):view(feature_size, 1)
local kappa_0 = 0.01
local nu_0 = feature_size
local lambda_0 = torch.eye(feature_size):cuda():mul(2)
-- torch.inverse(nu_0 * 
--   torch.CudaTensor({
--     {0.0366284242442, 0.00655439824317, 0.00223428470026, 0.00512935709104, 0.00706118920457, 0.00127367072023, 0.00895389914584, -0.00374665113559, 0.00494345676178, -0.0108999683881, },
--     {0.00655439824317, 0.0390891765439, -0.00130038432964, 0.00267932826372, 7.76895533872e-05, -0.000670595539531, -0.000162156024855, -0.00624318767337, 0.00433524044539, -0.000619171853721, },
--     {0.00223428470026, -0.00130038432964, 0.0446884605404, 0.00548076846674, -0.00285376214203, -0.00796014366193, 0.00940989578467, -0.00493952206002, 0.00139216856423, -0.00234435190168, },
--     {0.00512935709104, 0.00267932826372, 0.00548076846674, 0.0321686199303, 0.00346165276327, -0.0038337789042, 0.00530248525576, -0.0054437993534, -0.00404586384196, -0.00123830146687, },
--     {0.00706118920457, 7.76895533872e-05, -0.00285376214203, 0.00346165276327, 0.0417464946152, 0.00352988995952, 0.0125444622185, 0.00155953790617, 0.00150808524191, -0.00460548496398, },
--     {0.00127367072023, -0.000670595539531, -0.00796014366193, -0.0038337789042, 0.00352988995952, 0.0403215362515, -0.000409462230569, 0.00135282730524, 0.00191234159114, -9.72654600904e-05, },
--     {0.00895389914584, -0.000162156024855, 0.00940989578467, 0.00530248525576, 0.0125444622185, -0.000409462230569, 0.0403591606515, -0.00650039728275, -0.00601142474037, -0.00608422171382, },
--     {-0.00374665113559, -0.00624318767337, -0.00493952206002, -0.0054437993534, 0.00155953790617, 0.00135282730524, -0.00650039728275, 0.0449720012599, -0.00673620989614, -0.00177772886742, },
--     {0.00494345676178, 0.00433524044539, 0.00139216856423, -0.00404586384196, 0.00150808524191, 0.00191234159114, -0.00601142474037, -0.00673620989614, 0.0415150176486, 0.00106010428521, },
--     {-0.0108999683881, -0.000619171853721, -0.00234435190168, -0.00123830146687, -0.00460548496398, -9.72654600904e-05, -0.00608422171382, -0.00177772886742, 0.00106010428521, 0.0305230968794, },
--   })
-- )

--torch.diag(torch.Tensor(feature_size):fill(opt.lambda)):cuda()
-- local tmp = torch.eig(lambda_0:double(), 'N')[{{},{1}}]
-- local det_lambda_0 = torch.prod(tmp)
-- print(tmp, math.log(det_lambda_0))
det_lambda_0 = distributions.util.logdet(lambda_0:double())
print(det_lambda_0)

-- Get parameters
local theta, gradTheta = autoencoder:getParameters()

-- Create loss
local criterion = nn.BCECriterion() -- nn.MSECriterion()--
if opt.criterion == 'mse' then
  criterion = nn.MSECriterion()
end
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
  -- Reconstruction phase
  -- Forward propagation
  local xHat = autoencoder:forward(x) -- Reconstruction
  local loss = criterion:forward(xHat, x)
  -- Backpropagation
  local gradLoss = criterion:backward(xHat, x)
  autoencoder:backward(x, gradLoss)

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
  local z = normalized_encoder:forward(x)
  local z_d = z:double()
  means = {}
  covs = {}
  crp_init(z, means, covs, tables, mu_0, kappa_0, lambda_0, nu_0)
  -- print(z[{{1,5}}])
  -- print(z:size())
  
  --encoder:training()
  for ite = 1, opt.sample_times do
      --n_tables = gibbs_sample(z_d, belong, connect, tables, connected, n_tables, opt.alpha, mu_0, kappa_0, lambda_0, nu_0, det_lambda_0)
      crp_gibbs_sample(z_d, means, covs, tables, belong, 1e-6, mu_0:double(), kappa_0, lambda_0:double(), nu_0)
  end

  for ite = 1, opt.K do
      --n_tables = gibbs_sample(z_d, belong, connect, tables, connected, n_tables, opt.alpha, mu_0, kappa_0, lambda_0, nu_0, det_lambda_0)
      crp_gibbs_sample(z_d, means, covs, tables, belong, 1e-6, mu_0:double(), kappa_0, lambda_0:double(), nu_0)
      local l1_loss, l2_loss, S_loss, dz, S_dz = cal_gradient(z, connect, tables, opt.alpha, mu_0, kappa_0, lambda_0, nu_0, det_lambda_0)
      dz = dz * opt.delta / opt.K / n_points
      S_dz = S_dz * opt.gamma
      loss = loss + (l1_loss + l2_loss) * opt.delta / opt.K / n_points + S_loss * opt.gamma
      print(string.format("   [%s], loss1: %f, loss2: %f, S_loss: %f", 
        os.date("%c", os.time()), 
        l1_loss * opt.delta / opt.K / n_points, 
        l2_loss * opt.delta / opt.K / n_points,
        S_loss * opt.gamma)
      )

      normalized_encoder:backward(x, dz + S_dz)
  end

  return loss, gradTheta
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
    evaluate_cluster(tables, yTrain, epoch, loss, n_points, #label_unique)
  else
    print(string.format("  loss: %4f", loss[1]))
  end

  -- Plot training curve(s)
  local plots = {{'Autoencoder', torch.linspace(1, #losses, #losses), torch.Tensor(losses), '-'}}
  gnuplot.pngfigure('train_20_news.png')
  gnuplot.plot(table.unpack(plots))
  gnuplot.ylabel('Loss')
  gnuplot.xlabel('Batch #')
  gnuplot.plotflush()

  if epoch % 30 == 0 and opt.task == 'ddcrp' then
    torch.save('trained/' .. opt.name .. string.format('_%d_ddcrp', epoch), Model)
    torch.save(string.format("results/tables_at_epoch_%d", epoch), tables)
  end
end


-- Test
print('Testing')
x = XTest:narrow(1, 1, 10)
autoencoder:evaluate()
local xHat = autoencoder:forward(x)


-- Plot reconstructions
if opt.dataset == 'mnist' then
  image.save('Reconstructions.png', torch.cat(image.toDisplayTensor(x, 2, 10), image.toDisplayTensor(xHat, 2, 10), 1))
end


if opt.ddcrp == 'ddcrp' then
  normalized_encoder:forward(XTrain)
  torch.save(string.format('data/feature_learnt.t7'), normalized_encoder.output:double())
end
