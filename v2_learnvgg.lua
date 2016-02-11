require 'torch'
require 'nn'
require 'image'
require 'optim'
require 'lfs'
require 'io'
require 'loadcaffe'

--------------------------------------------------------------------------------

local cmd = torch.CmdLine()

-- Basic options
cmd:option('-train_dir', 'train/')
cmd:option('-test_dir', 'test/')
cmd:option('-test_template', 'test_template.csv')
cmd:option('-train_labels', 'train_labels.csv')
cmd:option('-content_size', 224, 'Size for content')
cmd:option('-style_size', 400, 'Size for style')
cmd:option('-gpu', 0, 'Zero-indexed ID of the GPU to use; for CPU mode set -gpu = -1')
cmd:option('-input_file', '')

-- Output options
cmd:option('-output_file', 'network.data')
cmd:option('-output_labels', 'test_results.csv')

-- Other options
cmd:option('-step_size', 1e-10)
cmd:option('-style_scale', 1.0)
cmd:option('-dropout', 0.5)
cmd:option('-num_strides', 1)
cmd:option('-save_iter', 2000)
cmd:option('-start_iter', 1)
cmd:option('-max_num_images', 45000)
cmd:option('-pooling', 'max', 'max|avg')
cmd:option('-proto_file', 'models/VGG_ILSVRC_19_layers_deploy.prototxt')
cmd:option('-model_file', 'models/VGG_ILSVRC_19_layers.caffemodel')
cmd:option('-backend', 'nn', 'nn|cudnn|clnn')
cmd:option('-seed', -1)

cmd:option('-content_layer', 'pool5', 'layer to learn from')
cmd:option('-style_layers', 'relu1_1,relu2_1,relu3_1,relu4_1', 'layers for style')


function nn.SpatialConvolutionMM:accGradParameters()
   -- nop.  not needed by our net
end

local function main(params)
   if params.gpu >= 0 then
      if params.backend ~= 'clnn' then
	 require 'cutorch'
	 require 'cunn'
	 cutorch.setDevice(params.gpu + 1)
      else
	 require 'clnn'
	 require 'cltorch'
	 cltorch.setDevice(params.gpu + 1)
      end
   else
      params.backend = 'nn'
   end

   if params.backend == 'cudnn' then
      require 'cudnn'
      if params.cudnn_autotune then
	 cudnn.benchmark = true
      end
      -- cudnn.SpatialConvolution.accGradParameters = nn.SpatialConvolutionMM.accGradParameters -- ie: nop
   end
   
   if params.seed >= 0 then
      torch.manualSeed(params.seed)
   end

   print('loadcaffe')
   local loadcaffe_backend = params.backend
   if params.backend == 'clnn' then loadcaffe_backend = 'nn' end
   local cnn = loadcaffe.load(params.proto_file, params.model_file, loadcaffe_backend):float()
   if params.gpu >= 0 then
      if params.backend ~= 'clnn' then
	 cnn:cuda()
      else
	 cnn:cl()
      end
   end


   print('loadlabels')
   -- load labels
   local flabels = assert(io.open(params.train_labels, "r"))
   local images = {}
   local fake_test = {}
   local i0 = 0
   while true do
      local line = flabels:read('*line')
      if line == nil or i0>100000 then
	 break
      end
      i0 = i0+1
      local parts = line:split('\n')[1]:split(';')
      if #parts == 2 and parts[1] ~= 'ID' then
	 parts[1] = params.train_dir..parts[1]..'.jpg'
	 parts[2] = tonumber(parts[2])
	 if #images >= params.max_num_images then
	    table.insert(fake_test, parts)
	 else
	    table.insert(images, parts)
	 end
      end
   end
   print("Training set: ", #images, #fake_test)
   flabels:close()

   print('loadtest')
   -- load test template
   flabels = assert(io.open(params.test_template, "r"))
   local test_images = {}
   i0 = 0
   while true do
      local line = flabels:read('*line')
      if line == nil or i0>100000 then
	 break
      end
      i0 = i0+1
      local parts = line:split('\n')[1]:split(';')
      if #parts == 2 and parts[1] ~= 'ID' then
	 table.insert(test_images, tonumber(parts[1]))
      end
   end
   print("Testing set: ", #test_images)
   flabels:close()

   -- Set up the network, inserting style descriptor modules
   
   local style_layers = params.style_layers:split(",")
   local pnormalize_features = params.normalize_features:split(",")
   local normalize_features = {}
   
   local qualitynet = nil
   local qualitylayers = nil
   if params.input_file ~= '' then
      print('loading network')
      obj = torch.load(params.input_file)
      qualitynet = obj.net
      obj = nil
   end
   
   print('reading caffe')
   local style_descrs = {}
   local content_descr = nil
   local next_style_idx = 1
   local features_net = nn.Sequential()
   local content_tocome = (qualitynet == nil)
   local content_net2 = nil
   if content_tocome then
      content_net2 = nn.Sequential()
   end
   for i = 1, #cnn do
      if content_tocome or next_style_idx <= #style_layers then
	 local layer = cnn:get(i)
	 local name = layer.name
	 local layer_type = torch.type(layer)
	 if next_style_idx <= #style_layers then
	    features_net:add(layer)
	 elseif content_tocome then
	    content_net2:add(layer)
	 end
	 if name == params.content_layer then
	    print("Setting up content layer  ", i, ":", layer.name)
	    content_tocome = false
	 end
	 if name == style_layers[next_style_idx] then
	    print("Setting up style layer  ", i, ":", layer.name, 'normalize:', norm)
	    table.insert(style_descrs, layer)
	    next_style_idx = next_style_idx + 1
	 end
      end
   end

   -- We don't need the base CNN anymore, so clean it up to save memory.
   cnn = nil
   for i=1, #features_net.modules do
      local module = features_net.modules[i]
      if torch.type(module) == 'nn.SpatialConvolutionMM' then
	 -- remove these, not used, but uses gpu memory
	 module.gradWeight = nil
	 module.gradBias = nil
      end
   end
   
   collectgarbage()
   
   local img_size = params.image_size
   local criterion = nn.MSECriterion()
   local output = torch.Tensor(1)
   if params.backend ~= 'clnn' then
      output = output:cuda()
      criterion:cuda()
   else
      output=output:cl()
   end
  
   if qualitynet == nil then
      local img = image.load(images[1][1], 3)
      local res = computeFeatures(params, img, features_net, style_descrs)
      qualitynet = buildNet(params, res, content_net2)
   end
   
   local errors = {}
   local batchSize = 10
   local offset = 0;
   local nextBatch = function ()
   end
   
   local x, dl_dx
   x, dl_dx = qualitynet:getParameters()
   
   local feval = function(x_new)
       -- copy the weight if are changed
       if x ~= x_new then
           x:copy(x_new)
       end
       local inputs, targets = nextBatch()
       dl_dx:zero()

       -- evaluate the loss function and its derivative with respect to x, given a mini batch
       local features = computeFeatures(params, style_descrs, features_net, inputs)
       local prediction = qualitynet:forward(features)
       local loss_x = criterion:forward(prediction, targets)
       
       qualitynet:backward(inputs, criterion:backward(prediction, targets))
       
       return loss_x, dl_dx
   end

   print('learning')
   local optim_params = {learningRate = 1e-4}
   for j = 1, params.num_strides do
      for i = 1, #images do
         _, fs = optim.adam(feval, x, optim_params)

         print('Image', i, 'error:', fs[1])

         if i % params.save_iter == 0 then
            torch.save(params.output_file, {net=qualitynet})
         end
      end
      torch.save(params.output_file, {net=qualitynet})
   end

   print('now testing')
   local fout = io.open('local_tests.csv', 'w')
   fout:write('ID,result,expected')
   if qualitynet~= nil then
      qualitynet:evaluate()
   end
   local toterror =0
   local output = torch.Tensor(1)
   for i=1, #fake_test do
      print('processing image '..i..': '..fake_test[i][1])
      collectgarbage()
 --   local freeMemory, totalMemory = cutorch.getMemoryUsage(params.gpu+1)
 --   print('Memory: ', freeMemory/1024/1024, 'MB free of ', totalMemory/1024/1024)
      local nimg = fake_test[i][1]
      local img = image.load(nimg, 3)

      if img then
	 local features = computeFeatures(params, style_descrs, features_net, inputs)
         local prediction = qualitynet:forward(features)
	 output[1] = fake_test[i][2]
         criterion:forward(prediction, output)
	 print('Note:', math.floor(prediction[1]+0.5), 'expected:', output[1], 'error:', math.floor(criterion.output+0.5))
	 toterror = toterror + criterion.output
	 fout:write('\n'..nimg..','..math.min(100, math.max(0, math.floor(prediction[1]+0.5)))..','..output[1]) 
      end
   end
   fout:close()
   print('\n\n\nExpected score:', toterror/(#fake_test), '\n\n\n')
   
   fout = io.open(params.output_labels, 'w')
   fout:write('ID;aesthetic_score')
   for i=1, #test_images do
      print('processing image '..i..': '..test_images[i])
      collectgarbage()
      local nimg = test_images[i]
      local fname = string.format('%s%07d.jpg', params.test_dir, nimg)
      local img = image.load(fname, 3)

      if img then
	 local features = get_features(params, style_descrs, features_net, inputs)
         local prediction = qualitynet:forward(features)
	 print(nimg, prediction[1])
	 fout:write('\r'..nimg..';'..math.min(100, math.max(0, math.floor(prediction[1]+0.5)))) 
      end
   end
   fout:close()
end

function shuffle(t)
   for i = #t, 2, -1 do
      local j = math.random(i)
      t[i], t[j] = t[j], t[i]
   end
end

function buildNet(params, res, content_net2)
   local qualitynet = nn.Sequential()
   -- Style
   local pstyle = nn.ParallelTable()
   local nEl = 0
   for i = 1, #(res[1]) do
      local j = nn.Sequential()
      if res[1][i]:size(1) > 256 then
         j:add(nn.SpatialConvolutionMM(res[1][i]:size(1), 256, 1, 1)
      end
      j:add(GramMatrix())
      j:add(nn.View(-1))
      nEl = nEl + res[1][i]:nElement()
      pstyle:add(j)
   end
   local lstyle = nn.Sequential()
   lstyle:add(pstyle)
   lstyle:add(nn.JoinTable(1))
   lstyle:add(nn.Linear(nEl, 512))
   -- Content
   local res2b = content_net2:forward(res[2])
   local lcontent = content_net2
   lcontent:add(nn.View(-1))
   lcontent:add(nn.Linear(res2b:nElement(), 512)
   -- Merge
   local merg = nn.ParallelTable()
   merg:add(lstyle)
   merg:add(lcontent)
   qualitynet:add(merg)
   qualitynet:add(nn.JoinTable(1))
   -- Quality Network
   qualitynet:add(nn.ReLU())
   qualitynet:add(nn.Dropout(params.dropout))
   qualitynet:add(nn.Linear(1024, 512))
   qualitynet:add(nn.ReLU())
   qualitynet:add(nn.Dropout(params.dropout))
   qualitynet:add(nn.Linear(512, 512))
   qualitynet:add(nn.ReLU())
   qualitynet:add(nn.Dropout(params.dropout))
   qualitynet:add(nn.Linear(512, 1))
   if params.gpu >= 0 then
      if params.backend ~= 'clnn' then
          qualitynet:cuda()
      else
          qualitynet:cl()
      end
   end
   return qualitynet, layers
end

function get_features(params, data, images, features_net, style_descrs)
   -- style features
   local imgs = torch.DoubleTensor(0, 3, params.style_size, params.style_size):zero()
   for i=1, #images do
      local im = image.scale(images[i], params.style_size, 'bilinear')
      local img = preprocess(im):float():view(1,img:size(1),img:size(2),img:size(3)):expand(1, 3, params.style_size, params.style_size)
      imgs:cat(img, 1)
   end
   if params.gpu >= 0 then
      if params.backend ~= 'clnn' then
	imgs = imgs:cuda()
      else
	imgs = imgs:cl()
      end
   end
   local freeMemory, totalMemory = cutorch.getMemoryUsage(params.gpu+1)
   print('Memory: ', freeMemory/1024/1024, 'MB free of ', totalMemory/1024/1024)
   style_net:forward(imgs)
   local res1 = {}
   for j, mod in ipairs(style_descrs) do
      table.insert(res1, mod.G)
   end
   -- content features
   imgs = torch.DoubleTensor(0, 3, params.style_size, params.style_size):zero()
   for i=1, #images do
      local im = image.scale(images[i], params.content_size, params.content_size, 'bilinear')
      local img = preprocess(im):float():view(1,img:size(1),img:size(2),img:size(3))
      imgs:cat(img, 1)
   end
   if params.gpu >= 0 then
      if params.backend ~= 'clnn' then
	imgs = imgs:cuda()
      else
	imgs = imgs:cl()
      end
   end
   local res2 = content_net:forward(imgs)
   
   return {res1, res2}
end


-- Preprocess an image before passing it to a Caffe model.
-- We need to rescale from [0, 1] to [0, 255], convert from RGB to BGR,
-- and subtract the mean pixel.
function preprocess(img)
   local mean_pixel = torch.DoubleTensor({103.939, 116.779, 123.68})
   local perm = torch.LongTensor{3, 2, 1}
   img = img:index(1, perm):mul(256.0)
   mean_pixel = mean_pixel:view(3, 1, 1):expandAs(img)
   img:add(-1, mean_pixel)
   return img
end


-- Returns a network that computes the CxC Gram matrix from inputs
-- of size C x H x W
function GramMatrix()
  local net = nn.Sequential()
  net:add(nn.View(-1):setNumInputDims(2))
  local concat = nn.ConcatTable()
  concat:add(nn.Identity())
  concat:add(nn.Identity())
  net:add(concat)
  net:add(nn.MM(false, true))
  return net
end

local params = cmd:parse(arg)
main(params)
