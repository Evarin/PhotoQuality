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
cmd:option('-train_labels', 'train_labels.csv')
cmd:option('-image_size', 400, 'Maximum height / width of generated image')
cmd:option('-gpu', 0, 'Zero-indexed ID of the GPU to use; for CPU mode set -gpu = -1')
cmd:option('-input_file', '')

-- Output options
cmd:option('-output_file', 'network.data')
cmd:option('-output_labels', 'test_results.csv')

-- Other options
cmd:option('-step_size', 1e-10)
cmd:option('-style_scale', 1.0)
cmd:option('-num_strides', 1)
cmd:option('-save_iter', 50001)
cmd:option('-max_num_images', 1000000)
cmd:option('-normalize_features', 'false')
cmd:option('-pooling', 'max', 'max|avg')
cmd:option('-proto_file', 'models/VGG_ILSVRC_19_layers_deploy.prototxt')
cmd:option('-model_file', 'models/VGG_ILSVRC_19_layers.caffemodel')
cmd:option('-backend', 'nn', 'nn|cudnn|clnn')
cmd:option('-seed', -1)

cmd:option('-style_layers', 'relu1_1,relu2_1,relu3_1,relu4_1,relu5_1', 'layers for style')


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
      cudnn.SpatialConvolution.accGradParameters = nn.SpatialConvolutionMM.accGradParameters -- ie: nop
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
	 table.insert(images, parts)
      end
      if #images > params.max_num_images then
	 break
      end
   end
   print("Training set: ", #images)
   flabels:close()

   -- Lists all the files from the test directory
   print('get test images')
   local test_images = {}
   for file in io.popen('find "'..params.test_dir..'" -maxdepth 1 -type f'):lines() do
      -- print("found file "..file)
      table.insert(test_images, file)
   end

   local style_layers = params.style_layers:split(",")
   local pnormalize_features = params.normalize_features:split(",")
   local normalize_features = {}

   -- Set up the network, inserting style descriptor modules
   print('reading caffe')
   local style_descrs = {}
   local next_style_idx = 1
   local net = nn.Sequential()
   for i = 1, #cnn do
      if next_style_idx <= #style_layers then
	 local layer = cnn:get(i)
	 local name = layer.name
	 local layer_type = torch.type(layer)
	 local is_pooling = (layer_type == 'cudnn.SpatialMaxPooling' or layer_type == 'nn.SpatialMaxPooling')
	 if is_pooling and params.pooling == 'avg' then
	    assert(layer.padW == 0 and layer.padH == 0)
	    local kW, kH = layer.kW, layer.kH
	    local dW, dH = layer.dW, layer.dH
	    local avg_pool_layer = nn.SpatialAveragePooling(kW, kH, dW, dH):float()
	    if params.gpu >= 0 then
	       if params.backend ~= 'clnn' then
		  avg_pool_layer:cuda()
	       else
		  avg_pool_layer:cl()
	       end
	    end
	    local msg = 'Replacing max pooling at layer %d with average pooling'
	    print(string.format(msg, i))
	    net:add(avg_pool_layer)
	 else
	    net:add(layer)
	 end
	 if name == style_layers[next_style_idx] then
	    local norm = (pnormalize_features[#pnormalize_features] == 'true')
	    if #pnormalize_features >= next_style_idx then
	       norm = (pnormalize_features[next_style_idx] == 'true')
	    end
	    table.insert(normalize_features, norm)
	    print("Setting up style layer  ", i, ":", layer.name, 'normalize:', norm)
	    local style_module = nn.StyleDescr(params.style_weight, norm):float()
	    if params.gpu >= 0 then
	       if params.backend ~= 'clnn' then
		  style_module:cuda()
	       else
		  style_module:cl()
	       end
	    end
	    net:add(style_module)
	    table.insert(style_descrs, style_module)
	    next_style_idx = next_style_idx + 1
	 end
      end
   end

   -- We don't need the base CNN anymore, so clean it up to save memory.
   cnn = nil
   for i=1,#net.modules do
      local module = net.modules[i]
      if torch.type(module) == 'nn.SpatialConvolutionMM' then
	 -- remove these, not used, but uses gpu memory
	 module.gradWeight = nil
	 module.gradBias = nil
      end
   end
   
   local qualitynet = nil
   if params.input_file ~= '' then
      print('loading network')
      obj = torch.load(params.input_file)
      qualitynet = obj.net
      if params.gpu >= 0 then
	 if params.backend ~= 'clnn' then
	    qualitynet:cuda()
	 else
	    qualitynet:cl()
	 end
      end
      obj = nil
   end
   
   collectgarbage()
   
   local img_size = params.image_size
   local criterion = nn.MSECriterion()
   local output = torch.Tensor(1)
   if params.backend ~= 'clnn' then
      output=output:cuda()
      criterion:cuda()
   else
      output=output:cl()
   end
  
   for j=1, params.num_strides do
      print("stride", j)
      shuffle(images)
      local toterror = 0
      for i=1, #images do
	 print('processing image '..i..': '..images[i][1])
	 collectgarbage()
	 local img = image.load(images[i][1], 3)
	 if img then
	    img = image.scale(img, img_size, 'bilinear')
	    local img_caffe = preprocess(img):float()
	    if params.gpu >= 0 then
	       if params.backend ~= 'clnn' then
		  img_caffe = img_caffe:cuda()
	       else
		  img_caffe = img_caffe:cl()
	       end
	    end
	    --local freeMemory, totalMemory = cutorch.getMemoryUsage(params.gpu+1)
	    --print('Memory: ', freeMemory/1024/1024, 'MB free of ', totalMemory/1024/1024)
	    net:forward(img_caffe)
	    local res = nil
	    for j, mod in ipairs(style_descrs) do
	       if j == 1 then
		  res = mod.G:clone()
	       else
		  res = res:cat(mod.G, 1)
	       end
	    end
	    -- init (with good input sizes)
	    if qualitynet == nil then
	       print(res:nElement())
	       print('Setting up network')
	       qualitynet = nn.Sequential()
	       qualitynet:add(nn.Linear(res:nElement(), 256))
	       qualitynet:add(nn.ReLU())
	       qualitynet:add(nn.Dropout(0.5))
	       qualitynet:add(nn.Linear(256, 256))
	       qualitynet:add(nn.ReLU())
	       qualitynet:add(nn.Dropout(0.5))
	       qualitynet:add(nn.Linear(256, 1))
	       if params.gpu >= 0 then
		  if params.backend ~= 'clnn' then
		     qualitynet:cuda()
		  else
		     qualitynet:cl()
		  end
	       end
	    end
	    
	    output[1] = images[i][2]
	    if params.backend ~= 'clnn' then
	       res:cuda()
	    else
	       res:cl()
	    end
	    
	    -- training
	    local fwd = qualitynet:forward(res)
	    criterion:forward(fwd, output)
	    print('Note:', math.floor(fwd[1]+0.5), 'expected:', output[1], 'error:', math.floor(criterion.output+0.5))
	    
	    toterror = toterror +  criterion.output
	    qualitynet:zeroGradParameters()
	    qualitynet:backward(res, criterion:backward(qualitynet.output, output))
	    qualitynet:updateParameters(params.step_size)
	    
	 end
	 if i % params.save_iter == 0 then
	   print('\n\n\nTotal error', toterror/params.save_iter, '\n\n\n')
	   print('saving network')
	   torch.save(params.output_file, {net=qualitynet})
	   toterror = 0
	 end
      end
      print('saving network')
     torch.save(params.output_file, {net=qualitynet})
   end

    local freeMemory, totalMemory = cutorch.getMemoryUsage(params.gpu+1)
    print('Memory: ', freeMemory/1024/1024, 'MB free of ', totalMemory/1024/1024)




   print('now testing')
   local fout = io.open(params.output_labels, 'w')
   fout:write('ID;aesthetic_score\n')
   for i=1, #test_images do
      print('processing image '..i..': '..test_images[i])
      collectgarbage()
 --   local freeMemory, totalMemory = cutorch.getMemoryUsage(params.gpu+1)
 --   print('Memory: ', freeMemory/1024/1024, 'MB free of ', totalMemory/1024/1024)
      local img = image.load(test_images[i], 3)
      local nimg = test_images[i]:split('/')
      nimg = nimg[#nimg]
      nimg = nimg:sub(1, #nimg-4)

      if img then
	 img = image.scale(img, img_size, 'bilinear')
	 local img_caffe = preprocess(img):float()
	 if params.gpu >= 0 then
	    if params.backend ~= 'clnn' then
	       img_caffe = img_caffe:cuda()
	    else
	       img_caffe = img_caffe:cl()
	    end
	 end
	 net:forward(img_caffe)
	 local res = nil
	 for j, mod in ipairs(style_descrs) do
	    if j == 1 then
	       res = mod.G:clone()
	    else
	       res = res:cat(mod.G, 1)
	    end
	 end
	 
	 local fwd = qualitynet:forward(res)
	 print(nimg, fwd[1])
	 fout:write(nimg..';'..math.min(100, math.max(0, math.floor(fwd[1]+0.5)))..'\r')
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
function GramMatrix(normalize)
  local net = nn.Sequential()
  net:add(nn.View(-1):setNumInputDims(2))
  if normalize then
    net:add(nn.Normalize(2))
  end
  local concat = nn.ConcatTable()
  concat:add(nn.Identity())
  concat:add(nn.Identity())
  net:add(concat)
  net:add(nn.MM(false, true))
  net:add(nn.View(-1))
  return net
end


-- Define an nn Module to compute style description (Gram Matrix) in-place
local StyleDescr, parent = torch.class('nn.StyleDescr', 'nn.Module')

function StyleDescr:__init(strength, normalize)
   parent.__init(self)
   self.strength = strength
   self.normalize = normalize or false
   
   self.gram = GramMatrix(self.normalize)
   self.G = nil
end

function StyleDescr:updateOutput(input)
   self.G = self.gram:forward(input)
   if not self.normalize then
     self.G:div(input:nElement())
   end
   self.output = input
   return self.output
end


local params = cmd:parse(arg)
main(params)
