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
]]

if opt.debug>0 then
  debugger = require('fb.debugger')
	debugger.enter()
end

local model_name = string.format('actor_T%d_lr%.6f_split%d', opt.T, opt.lr, opt.split)

print("creating model")
local cnn = torch.load(opt.cnn_path)
cnn:remove(#cnn.modules)
cnn:remove(#cnn.modules)
cnn:remove(#cnn.modules)
cnn:remove(#cnn.modules)
cnn:remove(#cnn.modules)
local loader = torch.load(string.format('./snapshots/%s_epoch%d.t7', model_name, opt.epoch))
local actor = loader.actor
local reward_list = loader.reward_list

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
local im_global_batch = torch.Tensor(opt.test_T, 1, 3, height, width)
local move_batch = torch.Tensor(opt.test_T, 1, 6)
local bb_batch = torch.Tensor(opt.test_T, 1, 4)

if opt.gpu > 0 then
	cnn = cnn:cuda()
	actor = actor:cuda()
	im_global_batch = im_global_batch:cuda()
	move_batch = move_batch:cuda()
	bb_batch = bb_batch:cuda()
end

-- test splits
local scene_names
if opt.split==1 then
	scene_names = {'Home_001_1','Home_001_2','Home_008_1'}
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
		init_correct = datasets[scene_id].annotations[image_id][object_id][5]
		init_score = datasets[scene_id].annotations[image_id][object_id][6]
		correct = init_correct
		score = init_score
		local w = bb[3]-bb[1]
		local h = bb[4]-bb[2]
		bb_batch[1][1][1] = (bb[1]+w/2)/width
		bb_batch[1][1][2] = (bb[2]+h/2)/height
		bb_batch[1][1][3] = w/width
		bb_batch[1][1][4] = h/height
		im_global_batch[1]:copy(input_im)
		local move_avail = datasets[scene_id].moves[image_id]  
		move_batch[1]:copy(datasets[scene_id].moves[image_id]:gt(0))

		print(string.format('Episodes %d: object_id %d: ', idx, object_id))
		print(string.format('image_id: %d, correct: %d, score: %.4f', image_id, init_correct, init_score))
		results[idx][{{},{1}}] = object_id
		results[idx][1][2] = image_id
		results[idx][1][3] = 0
		results[idx][1][4] = init_correct
		results[idx][1][5] = init_score
		results[idx][1][{{6,9}}] = bb

		local next_im = input_im
		local next_bb = bb
		local next_move_avail = move_avail
		local actions = torch.Tensor(opt.test_T)
		local probs = torch.Tensor(opt.test_T, 6):fill(0):cuda()
		local conv_feat = torch.Tensor(opt.test_T,1,128,27,48):cuda()
		local last_t 
		for t=1,opt.test_T do
			last_t = t
			cnn:evaluate()
			actor:evaluate()
			-- conv_feat[t] = -- cnn:forward(im_global_batch[t]):clone()
			-- probs[t] = actor:forward({conv_feat[t],move_batch[t],bb_batch[t]})
		
			--random baseline
			-- actions[t] = torch.random(6)
			
			-- forward baseline
			--actions[t] = 1
			
			-- RL
			-- _,action = probs[t]:max(1)
			-- actions[t] = action[1]

			-- Greedy
			
			local wc = (next_bb[1] + next_bb[3]) * opt.scale / 2
			local w1, w2 = width / 4, width * 3 / 4

			actions[t] = 1
			if (wc < w1) then
   		  actions[t] = 6
   		end
   		if (wc > w2) then
   		  actions[t] = 5
   		end
   		
   		-- print('w', wc, w1, w2)
   		-- print('bb', next_bb[1], next_bb[3])

			-- take the action
			results[idx][t+1][3] = actions[t]
			local next_image_id = next_move_avail[actions[t]]
			if next_image_id > 0 then
				next_im = datasets[scene_id].images[next_image_id]
				next_im = transform(next_im)
				next_bb = datasets[scene_id].annotations[next_image_id][object_id][{{1,4}}]
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
				im_global_batch[t+1][1]:copy(next_im)
				move_batch[t+1][1]:copy(next_move_avail:gt(0))
				local w = next_bb[3]-next_bb[1]
				local h = next_bb[4]-next_bb[2]
				bb_batch[t+1][1][1] = (next_bb[1]+w/2)/width
				bb_batch[t+1][1][2] = (next_bb[2]+h/2)/height
				bb_batch[t+1][1][3] = w/width
				bb_batch[t+1][1][4] = h/height
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
	print(string.format('[REINFORCE] Accuracy: %d/%d(%.3f)', 
				total_correct[scene_id], n_trains[scene_id], total_correct[scene_id]/n_trains[scene_id]))
	print(string.format('[REINFORCE] Average score of correct: %.3f', total_score[scene_id]/total_correct[scene_id]))
	print(string.format('[Baseline 1] Accuracy: %d/%d(%.3f)', 
				total_init_correct[scene_id], n_trains[scene_id], total_init_correct[scene_id]/n_trains[scene_id]))
	print(string.format('[Baseline 1] Average score of correct: %.3f', total_init_score[scene_id]/total_init_correct[scene_id]))
end
