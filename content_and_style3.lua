require 'torch'
require 'nn'
require 'image'
require 'optim'
require 'lfs'
require 'io'

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
cmd:option('-batch_size', 5)
cmd:option('-max_num_images', 45000)
cmd:option('-pooling', 'max', 'max|avg')
cmd:option('-model_file', 'content.data')
cmd:option('-backend', 'nn', 'nn|cudnn|clnn')
cmd:option('-seed', -1)
cmd:option('-print_memory', false)
cmd:option('-flip', false)

cmd:option('-content_layer', 37, 'layer to learn from') -- pool5
cmd:option('-style_layers', '2,7', 'layers for style') -- relu1_1, relu2_1


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

   print('loadbasenet')
   local obj = torch.load(params.model_file)
   local cnn = obj.net:get(1)
   cnn:clearState()
   obj = nil
   
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
   
   local qualitynet = nil
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
   local style_net = nn.Sequential()
   local content_net = nn.Sequential()
   local content_net2 = nil
   if qualitynet == nil then
      content_net2 = nn.Sequential()
   end
   local content_tocome = (qualitynet == nil)
   print(#cnn)
   for i = 1, #cnn do
      if content_tocome or next_style_idx <= #style_layers then
	 local layer = cnn:get(i)
	 local name = layer.name
	 print(i)
	 local layer_type = torch.type(layer)
	 if next_style_idx <= #style_layers then
	    style_net:add(layer)
	    content_net:add(layer)
	 elseif content_tocome then
	    content_net2:add(layer)
	 end
	 if i == params.content_layer then
	    print("Setting up content layer  ", i, ":", layer.name)
	    content_tocome = false
	 end
	 if i == tonumber(style_layers[next_style_idx]) then
	    print("Setting up style layer  ", i, ":", layer.name)
	    local nlayer = nn.StyleDescr(false)
	    if params.backend ~= 'clnn' then
		nlayer:cuda()
	    else
		nlayer:cl()
	    end
	    style_net:add(nlayer)
	    table.insert(style_descrs, nlayer)
	    next_style_idx = next_style_idx + 1
	 end
      end
   end

   -- We don't need the base CNN anymore, so clean it up to save memory.
   cnn = nil
   for i=1, #content_net.modules do
      local module = content_net.modules[i]
      if torch.type(module) == 'nn.SpatialConvolutionMM' then
	 -- remove these, not used, but uses gpu memory
	 module.gradWeight = nil
	 module.gradBias = nil
      end
   end
   
   collectgarbage()
   
   local img_size = params.image_size
   local criterion = nn.MSECriterion()
  
   if qualitynet == nil then
      local img = image.load(images[1][1], 3)
      local res = computeFeatures(params, style_net, style_descrs, content_net, img)
      qualitynet = buildNet(params, res, content_net2)
   else
      for i=1, #qualitynet.modules do
         local module = qualitynet.modules[i]
         if torch.type(module) == 'nn.Dropout' then
             module:setp(params.dropout)
         end
      end
   end
   
   local target = torch.Tensor(1)
   if params.backend ~= 'clnn' then
      target = target:cuda()
      criterion:cuda()
   else
      target = target:cl()
      criterion:cuda()
   end
   local offset = 1
   local totcount = 0
   local trainorder = torch.randperm(#images)
   local x, dl_dx
   x, dl_dx = qualitynet:getParameters()
   
   local feval = function(x_new)
       -- copy the weight if are changed
       collectgarbage()
       if x ~= x_new then
           x:copy(x_new)
       end
       
       if offset > #images then
          trainorder = torch.randperm(#images)
	  offset = 1
       end
       
       local indices = trainorder[offset]
       local fname = images[indices][1]
       local iscore = images[indices][2]

       local img = image.load(fname, 3)
       if params.flip and math.random(2) == 1 then
	  img = image.hflip(img)
       end
       target[1] = iscore
       offset = offset + 1
       dl_dx:zero()

       -- evaluate the loss function and its derivative with respect to x
       local features = computeFeatures(params, style_net, style_descrs, content_net, img)

       local prediction = qualitynet:forward(features)

       print('prediction', math.ceil(prediction[1]), 'objective',  target[1])

       if params.print_memory then
	  local freeMemory, totalMemory = cutorch.getMemoryUsage(params.gpu+1)
	  print('Memory: ', freeMemory/1024/1024, 'MB free of ', totalMemory/1024/1024)
       end

       local loss_x = criterion:forward(prediction, target)
       
       qualitynet:backward(features, criterion:backward(prediction, target))

       if params.print_memory then
	  local freeMemory, totalMemory = cutorch.getMemoryUsage(params.gpu+1)
	  print('Memory: ', freeMemory/1024/1024, 'MB free of ', totalMemory/1024/1024)
       end
       
       return loss_x, dl_dx
   end

   print('learning')
   local optim_params = {learningRate = params.step_size}
   local niter = 1
   local toterror = 0
   local errors = {}
   while totcount <= #images * params.num_strides do
      _, fs = optim.adam(feval, x, optim_params)

      print('Iteration '..niter, '                       error:', math.floor(fs[1]+0.5))
      toterror = toterror + fs[1]
      totcount = totcount + 1

      if niter % params.save_iter == 0 then
	 print(errors)
	 print('Total error', toterror/params.save_iter)
	 table.insert(errors, toterror/params.save_iter)
	 toterror = 0
	 qualitynet:clearState()
         torch.save(params.output_file, {net=qualitynet})
      end
      niter = niter + 1
   end
   if params.num_strides > 0 then
      qualitynet:clearState()
      torch.save(params.output_file, {net=qualitynet})
   end

   print('now testing')
   local fout = io.open('local_tests.csv', 'w')
   fout:write('ID,result,expected')
   if qualitynet~= nil then
      qualitynet:evaluate()
      for i=1, #qualitynet.modules do
         local module = qualitynet.modules[i]
         if torch.type(module) == 'nn.Dropout' or torch.type(module) == 'nn.Dropout2' then
             module:setp(0)
         end
      end
   end
   local toterror =0
   local output = torch.Tensor(1)
   output = output:cuda()
   for i=1, #fake_test do
      print('processing image '..i..': '..fake_test[i][1])
      collectgarbage()
      if params.print_memory then
	 local freeMemory, totalMemory = cutorch.getMemoryUsage(params.gpu+1)
	 print('Memory: ', freeMemory/1024/1024, 'MB free of ', totalMemory/1024/1024)
      end
      local nimg = fake_test[i][1]
      local img = image.load(nimg, 3)

      if img then
	 local features = computeFeatures(params, style_net, style_descrs, content_net, img)
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
	 local features = computeFeatures(params, style_net, style_descrs, content_net, img)
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
      local sz = res[1][i]:size(1)
      j:add(nn.View(sz*sz))
      nEl = nEl + sz*sz
      pstyle:add(j)
   end
   print(nEl)
   local lstyle = nn.Sequential()
   lstyle:add(pstyle)
   lstyle:add(nn.JoinTable(1, 1))
   lstyle:add(nn.Linear(nEl, 512))
   -- Content
   local lcontent = nn.Sequential()
   local res2b = content_net2:forward(res[2])
   lcontent:add(content_net2)
   lcontent:add(nn.View(res2b:nElement()))
   lcontent:add(nn.Linear(res2b:nElement(), 1024))
   -- Merge
   local merg = nn.ParallelTable()
   merg:add(lstyle)
   merg:add(lcontent)
   qualitynet:add(merg)
   qualitynet:add(nn.JoinTable(1, 1))
   -- Quality Network
   qualitynet:add(nn.ReLU())
   qualitynet:add(nn.Dropout(params.dropout))
   qualitynet:add(nn.Linear(1536, 1024))
   qualitynet:add(nn.ReLU())
   qualitynet:add(nn.Dropout(params.dropout))
   qualitynet:add(nn.Linear(1024, 512))
   qualitynet:add(nn.ReLU())
   qualitynet:add(nn.Dropout(params.dropout))
   qualitynet:add(nn.Linear(512, 1))
   qualitynet:add(nn.ReLu())
   if params.gpu >= 0 then
      if params.backend ~= 'clnn' then
          qualitynet:cuda()
      else
          qualitynet:cl()
      end
   end
   return qualitynet
end

function computeFeatures(params, style_net, style_descrs, content_net, imag)
   -- style features
   local imgs
   local im = image.scale(imag, params.style_size, 'bilinear')
   imgs = preprocess(im):float()
   if params.gpu >= 0 then
      if params.backend ~= 'clnn' then
	imgs = imgs:cuda()
      else
	imgs = imgs:cl()
      end
   end
   if params.print_memory then
      local freeMemory, totalMemory = cutorch.getMemoryUsage(params.gpu+1)
      print('Memory: ', freeMemory/1024/1024, 'MB free of ', totalMemory/1024/1024)
   end
   style_net:forward(imgs)
   local res1 = {}
   for j, mod in ipairs(style_descrs) do
      table.insert(res1, mod.G)
   end
   -- content features
   im = image.scale(imag, params.content_size, params.content_size, 'bilinear')
   imgs = preprocess(im):float()
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
function GramMatrix(batchSize)
  local net = nn.Sequential()
  net:add(nn.View(-1):setNumInputDims(2))
--  net:add(nn.Normalize(2))
  local concat = nn.ConcatTable()
  concat:add(nn.Identity())
  concat:add(nn.Identity())
  net:add(concat)
  net:add(nn.MM(false, true))
  return net
end


-- Define an nn Module to compute style description (Gram Matrix) in-place
local StyleDescr, parent = torch.class('nn.StyleDescr', 'nn.Module')

function StyleDescr:__init(cgrad)
   parent.__init(self)
   self.strength = 1
   self.cgrad = cgrad or false
   
   self.gram = GramMatrix()
   self.G = nil
end

function StyleDescr:updateOutput(input)
   self.G = torch.triu(self.gram:forward(input))
   self.G:div(input:nElement())

   self.output = input
   return self.output
end

function StyleDescr:updateGradInput(input, gradOutput)
  if not self.cgrad then
    self.gradInput = self.gradOutput
    return self.gradInput
  end
  self.gradInput = self.gram:backward(input, gradOutput)
  self.gradInput:div(input:nElement())
  self.gradInput:mul(self.strength)
  return self.gradInput
end


local Dropout2, Parent = torch.class('nn.Dropout2', 'nn.Dropout')

function Dropout2:__init(p,v1,inplace)
   Parent.__init(self,p,v1,inplace)
end

function Dropout2:updateOutput(input)
   if self.inplace then
      self.output = input
   else
      self.output:resizeAs(input):copy(input)
   end
   if self.p > 0 then
      if self.train then
         self.noise:resizeAs(input)
         self.noise:bernoulli(1-self.p)
         self.output:cmul(self.noise)
      end
   end
   return self.output
end

function Dropout2:updateGradInput(input, gradOutput)
   if self.train then
      if self.inplace then
         self.gradInput = gradOutput
      else
         self.gradInput:resizeAs(gradOutput):copy(gradOutput)
      end
      if self.p > 0 then
         self.gradInput:cmul(self.noise) -- simply mask the gradients with the noise vector
      end
   else
      if self.inplace then
         self.gradInput = gradOutput
      else
         self.gradInput:resizeAs(gradOutput):copy(gradOutput)
      end
   end
   return self.gradInput
end


local params = cmd:parse(arg)
main(params)
