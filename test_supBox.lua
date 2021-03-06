require 'torch'
require 'nn'
require 'nngraph'
require 'optim'
require 'cunn'
require 'cudnn'
require 'gnuplot'
t = require './fb.resnet.torch/datasets/transforms'
torch.setdefaulttensortype('torch.FloatTensor')

opt = lapp[[
  --img_scale         (default 160)
  --lr								(default 0.00005)
  --epoch							(default 30)
  --scale							(default 0.2)
  --threshold					(default 0.9)
	--T									(default 5)
	--test_T						(default 5)
	--split							(default 1)
	--cnn_path					(default './snapshots/resnet-18.t7')
  -g, --gpu           (default 1)
	-d, --debug					(default 0)
	--loadModel        (default '')
]]

if opt.debug>0 then
  debugger = require('fb.debugger')
	debugger.enter()
end

local model_name = string.format('actor_T%d_lr%.6f_split%d', opt.T, opt.lr, opt.split)

print("creating model")
local cnn = torch.load(opt.loadModel).cnn
-- mean subtraction, data augmentation
meanstd = {
   mean = { 0.485, 0.456, 0.406 },
   std = { 0.229, 0.224, 0.225 },
}
transform = t.Compose{
	t.ColorNormalize(meanstd),
}

local height = 1080*opt.scale
local width = 1920*opt.scale

if opt.gpu > 0 then
	cnn = cnn:cuda()
end

-- test splits
local scene_names
if opt.split==1 then
	scene_names = {'Home_001_1', 'Home_001_2','Home_008_1'}
elseif opt.split==2 then
	-- scene_names = {'Home_003_1','Home_003_2','Office_001_1'}
	scene_names = {'Home_003_1','Home_003_2'}
else
	scene_names = {'Home_002_1','Home_014_1','Home_014_2'}
end

-- loading testing sets
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

--testing
local total_correct = torch.Tensor(n_scene):fill(0)
local total_score = torch.Tensor(n_scene):fill(0)
local total_init_correct = torch.Tensor(n_scene):fill(0)
local total_init_score = torch.Tensor(n_scene):fill(0)
for scene_id=1,n_scene do
	local results = torch.Tensor(n_trains[scene_id],opt.test_T+1,9):fill(0)
	for idx=1,n_trains[scene_id] do
		-- initial correct and score
		local init_correct = 0
		local init_score = 0
		local correct = 0
		local score = 0

		local image_id = datasets[scene_id].candidates[idx][1]
		local object_id = datasets[scene_id].candidates[idx][2]
		local input_im = datasets[scene_id].images[image_id]
		input_im = transform(input_im)
		local bb = datasets[scene_id].annotations[image_id][object_id][{{1,4}}]
    bb = (bb * opt.scale):int()
		local mask = torch.zeros(1, height, width)
    if bb[1] == 0 then
		  bb[1] = 1
		end
	  if bb[2] == 0 then
	    bb[2] = 1
		end
		if bb[3] <= bb[1] then
	    bb[3] = bb[1] + 1
	  end
		if bb[4] <= bb[2] then
		  bb[4] = bb[2] + 1
		end
		
		local w = bb[3]-bb[1]
		local h = bb[4]-bb[2]
		mask:sub(1, 1, bb[2] + 1, bb[4], bb[1] + 1, bb[3]):copy((torch.ones(1, h, w) * 1):clone())


		init_correct = datasets[scene_id].annotations[image_id][object_id][5]
		init_score = datasets[scene_id].annotations[image_id][object_id][6]
		correct = init_correct
		score = init_score
		local w = bb[3]-bb[1]
		local h = bb[4]-bb[2]

		local next_im = input_im
		local next_mask = torch.zeros(1, height, width)
		local next_bb = bb
		local move_avail = datasets[scene_id].moves[image_id]  
    local next_move_avail = move_avail
    local actions = torch.Tensor(opt.test_T)
    
		print(string.format('Episodes %d: object_id %d: ', idx, object_id))
		print(string.format('image_id: %d, correct: %d, score: %.4f', image_id, init_correct, init_score))
		results[idx][{{},{1}}] = object_id
		results[idx][1][2] = image_id
		results[idx][1][3] = 0
		results[idx][1][4] = init_correct
		results[idx][1][5] = init_score
		results[idx][1][{{6,9}}] = bb
    
    local input = torch.Tensor(1, 4, height, width)
    input[1]:sub(1, 3):copy(input_im)
    input[1]:sub(4, 4):copy(mask)
		for t=1,opt.test_T do
			cnn:evaluate()
			input = input:cuda()
			local output = cnn:forward(input)
			local _, pred = output[1]:max(1)
			local next_image_id = image_id
			
			pred = pred[1]
			-- print('pred', pred)
			-- print('pred', pred, pred:float()[1][1])

			if pred > 6.5 then
			  actions[t] = 1
			else
			  actions[t] = pred
			end
			-- random baseline
			--actions[t] = torch.random(6)
			
			-- forward baseline
			--actions[t] = 1
			while next_move_avail[actions[t]] == 0 do
			  actions[t] = torch.random(6)
			end
			next_image_id = next_move_avail[actions[t]]
			
			if t == 1 then
			  next_image_id = image_id
			end
			
			if next_image_id > 0 then
				next_im = datasets[scene_id].images[next_image_id]
				next_im = transform(next_im)
				next_bb = datasets[scene_id].annotations[next_image_id][object_id][{{1,4}}]
				next_bb = (next_bb * opt.scale):int()

				next_mask = torch.zeros(1, height, width)
		    if next_bb[1] == 0 then
				  next_bb[1] = 1
				end
			  if next_bb[2] == 0 then
			    next_bb[2] = 1
				end
				if next_bb[3] <= next_bb[1] then
			    next_bb[3] = next_bb[1] + 1
			  end
				if next_bb[4] <= next_bb[2] then
				  next_bb[4] = next_bb[2] + 1
				end
				local w = next_bb[3]-next_bb[1]
				local h = next_bb[4]-next_bb[2]  
				next_mask:sub(1, 1, next_bb[2] + 1, next_bb[4], next_bb[1] + 1, next_bb[3]):copy((torch.ones(1, h, w) * 1):clone())
				next_move_avail = datasets[scene_id].moves[next_image_id]  
				correct = datasets[scene_id].annotations[next_image_id][object_id][5]
				score = datasets[scene_id].annotations[next_image_id][object_id][6]
				print(string.format('action: %d, image_id: %d, correct: %d, score: %.4f, bb(%d,%d,%d,%d)', actions[t], next_image_id, correct, score, next_bb[1],next_bb[2],next_bb[3],next_bb[4]))

				results[idx][t+1][2] = next_image_id
				results[idx][t+1][4] = correct
				results[idx][t+1][5] = score
				results[idx][t+1][{{6,9}}] = next_bb
			else
				print(string.format('action: %d(unavailable!), image_id: %d, correct: %d, score: %.4f, bb(%d,%d,%d,%d)', actions[t], next_image_id, correct, score, next_bb[1],next_bb[2],next_bb[3],next_bb[4]))
				results[idx][t+1][2] = results[idx][t][2]
				results[idx][t+1][4] = results[idx][t][4]
				results[idx][t+1][5] = results[idx][t][5]
				results[idx][t+1][{{6,9}}] = results[idx][t][{{6,9}}]
			end
			if score >= opt.threshold then
				print('FOUND!')
				break
			end
			-- prepare data for next time step
			if t < opt.test_T then
				input[1]:sub(1, 3):copy(next_im)
				input[1]:sub(4, 4):copy(next_mask)

				local w = next_bb[3]-next_bb[1]
				local h = next_bb[4]-next_bb[2]
			end
		end

		total_correct[scene_id] = total_correct[scene_id] + correct
		total_init_correct[scene_id] = total_init_correct[scene_id] + init_correct
		if correct == 1 then
			total_score[scene_id] = total_score[scene_id] + score
			total_init_score[scene_id] = total_init_score[scene_id] + init_score
		end
	end
	--torch.save(string.format('actor_results_%s.t7',scene_names[scene_id]),results)
end

for scene_id=1,n_scene do
	print('Statistics: ' .. scene_names[scene_id])
	print(string.format('[Sup] Accuracy: %d/%d(%.3f)', 
				total_correct[scene_id], n_trains[scene_id], total_correct[scene_id]/n_trains[scene_id]))
	print(string.format('[Sup] Average score of correct: %.3f', total_score[scene_id]/total_correct[scene_id]))
	print(string.format('[Baseline 1] Accuracy: %d/%d(%.3f)', 
				total_init_correct[scene_id], n_trains[scene_id], total_init_correct[scene_id]/n_trains[scene_id]))
	print(string.format('[Baseline 1] Average score of correct: %.3f', total_init_score[scene_id]/total_init_correct[scene_id]))
end
