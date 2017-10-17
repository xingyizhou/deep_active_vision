require 'torch'
require 'nn'
require 'optim'
require 'cunn'
require 'cudnn'
require 'gnuplot'
local model_utils = require 'util.model_utils'
local t = require './fb.resnet.torch/datasets/transforms'
local create_model = require 'create_model'
torch.setdefaulttensortype('torch.FloatTensor')

opt = lapp[[
  --batch_size				(default 20)
	--iter_per_epoch		(default 1000)
  --lr								(default 0.00005)
  --max_epoch					(default 30)
  --scale							(default 0.2)
  --threshold					(default 0.9)
	--T									(default 5)
	--split							(default 1)
	--cnn_path					(default './snapshots/resnet-18.t7')
  -g, --gpu           (default 1)
	-d, --debug					(default 0)
]]

if opt.debug>0 then
  debugger = require('fb.debugger')
	debugger.enter()
end

local model_name = string.format('supImg_T%d_lr%.6f_split%d', opt.T, opt.lr, opt.split)

local timer = torch.Timer()
print("creating model")
local cnn = torch.load(opt.cnn_path)
cnn:remove(#cnn.modules)
cnn:remove(#cnn.modules)
cnn:remove(#cnn.modules)
cnn:add(nn.SpatialAveragePooling(12, 7, 1, 1))
cnn:add(nn.View(512):setNumInputDims(3))
cnn:add(nn.Linear(512, 7))

local criterion = nn.CrossEntropyCriterion()
criterion = criterion:cuda()

local config = {
	learningRate = opt.lr,
	beta1 = 0.9,
	beta2 = 0.999,
}



local state = {}
local reward_list = torch.Tensor(1,1):fill(0)

-- mean subtraction, data augmentation
local meanstd = {
   mean = { 0.485, 0.456, 0.406 },
   std = { 0.229, 0.224, 0.225 },
}
local transform = t.Compose{
	t.ColorNormalize(meanstd),
}

local height = 1080*opt.scale
local width = 1920*opt.scale
-- local im_global_batch = torch.Tensor(opt.T, 1, 3, height, width)

if opt.gpu > 0 then
	cnn = cnn:cuda()
end

local params, grad_params = cnn:getParameters()
grad_params:zero()

-- train splits
local scene_names
if opt.split==1 then
	scene_names = {'Home_002_1','Home_003_1','Home_003_2','Home_004_1','Home_004_2','Home_005_1','Home_005_2','Home_006_1','Home_014_1','Home_014_2'}--,'Office_001_1'}
elseif opt.split==2 then
	scene_names = {'Home_001_1','Home_001_2','Home_002_1','Home_004_1','Home_004_2','Home_005_1','Home_005_2','Home_006_1','Home_008_1','Home_014_1','Home_014_2'}
else
	scene_names = {'Home_001_1','Home_001_2','Home_003_1','Home_003_2','Home_004_1','Home_004_2','Home_005_1','Home_005_2','Home_006_1','Home_008_1'}--,'Office_001_1'}
end

-- loading trainsets
local datasets = {}
local n_trains = torch.Tensor(#scene_names)
for i=1,#scene_names do
	print('loading training images... ' .. scene_names[i])
	datasets[i] = torch.load(string.format('data/rohit_%s.t7',scene_names[i]))
	-- image scale [0,255] -> [0,1]
	datasets[i].images = datasets[i].images:float():div(255)
	n_trains[i] = datasets[i].candidates:size(1)
end
local n_scene = #scene_names

for e=1,opt.max_epoch do
	for i=1,opt.iter_per_epoch do
		-- prepare the image batch
		timer:reset(); timer:resume()
		local input = torch.Tensor(opt.batch_size, 3, height, width)
		local target = torch.Tensor(opt.batch_size, 1)
		for b=1,opt.batch_size do
		  -- initial correct and score
		  local init_correct = 0
		  local init_score = 0
		  local correct = 0
		  local score = 0
		  local scene_id = torch.random(n_scene)
		  local train_idx = torch.random(n_trains[scene_id])
		  local image_id = datasets[scene_id].candidates[train_idx][1]
		  local object_id = datasets[scene_id].candidates[train_idx][2]
		  local input_im = datasets[scene_id].images[image_id]
		  input_im = transform(input_im)
		  local bb = datasets[scene_id].annotations[image_id][object_id][{{1,4}}]
		  local w = bb[3]-bb[1]
		  local h = bb[4]-bb[2]

		  correct = datasets[scene_id].annotations[image_id][object_id][5]
		  score = datasets[scene_id].annotations[image_id][object_id][6]
		  local move_avail = datasets[scene_id].moves[image_id]  
		  
		  local bestScore = score
		  local bestDir = 7
		  for dir = 1, 6 do 
		    local dirId = move_avail[dir]
		    if dirId > 0 then
		      dirScore = datasets[scene_id].annotations[dirId][object_id][6]
		      if dirScore > bestScore then
		        bestScore = dirScore
		        bestDir = dir
		      end
		    end
		  end
		  input[b]:copy(input_im)
		  target[b] = bestDir
		end --for b=1,opt.batch_size do
    
    input = input:cuda()
    target = target:cuda()

    
    local output = cnn:forward(input)
    local loss = criterion:forward(output, target)
    cnn:zeroGradParameters()
    criterion:backward(output, target)
    cnn:backward(input, criterion.gradInput)
    
    local _, pred = output:float():topk(1)
    local correct = pred:eq(target:long():view(opt.batch_size, 1):expandAs(pred))
    local acc = (correct:narrow(2, 1, 1):sum() / opt.batch_size)
    --[[
    local nCorrect = 0
    for b = 1, opt.batch_size do
      -- print('pred, target' , pred[b] , ' ' , target[b])
      if (pred[b] == target[b]) then
        print('!!!')
        nCorrect = nCorrect + 1
      end
    end 
    --]]
		-- update gradients
		local feval = function(x)
			collectgarbage()
			return err, grad_params
		end
		optim.adam(feval, params, config)

		grad_params:zero()
		timer:stop()
    
    if i % 100 == 0 then
  		print("epoch: ", e .. " iter: ", i, '/', opt.iter_per_epoch, 'loss ', loss, 'Acc ', acc)
    end
	end
	
	if e%10 == 0 then
		snapshot = {}
		snapshot.cnn = cnn
		snapshot.state = state
		snapshot.config = config
		snapshot.reward_list = torch.Tensor(reward_list)
		torch.save(string.format('./snapshots/SupImg_%s_epoch%d.t7', model_name, e),snapshot)
	end
end

