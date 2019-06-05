-- testing script
--
-- Wenjie Luo
-- David-Alexandre Beaupre
--

require 'xlua'
require 'optim'
require 'cunn'
require 'gnuplot'
require 'io'

require 'MulClassNLLCriterion'
require 'TestDataHandler'
local c = require 'trepl.colorize'
lapp = require 'pl.lapp'
opt = lapp[[
    --folder_name              (default '')
    --dual_net                 (default 1)
    --data_version             (default "kitti2015")
    -g, --gpuid                (default 0)                  gpu id
    --test_num                   (default 10)               test images
    --model                    (default dot_win37_dep9)     model name
    --data_root                (default "/ais/gobi3/datasets/kitti/scene_flow/training") dataset root folder
    --util_root                (default "")                 dataset root folder
    --tb                       (default 100)                test batch size
    --type                     (default 'normal')           normal, small, tiny (size of dataset used)
    --invert                   (default 0)                  switch lwir-rgb images
    --share                    (default 1)                  share params or not

    --psz                      (default 9)                  half width
    --half_range               (default 100)                half range
]]

half_range = opt.half_range
model_name = opt.model
model_params = opt.folder_name .. '/param_epoch_200.t7'
model_bnmeanstd = opt.folder_name .. '/bn_meanvar_epoch_200.t7'

print(opt)

print(c.blue '==>' ..' configuring model')

torch.manualSeed(123)
cutorch.setDevice(opt.gpuid+1)
torch.setdefaulttensortype('torch.FloatTensor')

extension = '.jpg'
val_bin = 'gab_test'
offset = 0
if opt.data_version == 'stcharles' then
    extension = '.png'
    val_bin = 'plsc_test'
elseif opt.data_version == 'all' then
    extension = '.png'
    test_bin = 'all_test'
    offset = 1012 -- 1 = 1012, 2 = 940, 3 = 984
end

data_type = opt.type

print(c.blue '==>' ..' loading data')

my_dataset = TestDataHandler(opt.data_root, opt.util_root, opt.test_num, opt.psz, opt.half_range, 1, test_bin, extension, data_type, offset, opt.invert)

local function load_model_dual()
    require('models/' .. model_name .. '.lua')
    model = nn.Sequential()
    model_p = nn.ParallelTable()
    l_model = create_model(half_range*2 + 1, 3):cuda()
    r_model = create_model(half_range*2 + 1, 3):cuda()
    if opt.share == 0 then
        r_model = l_model:clone('weight','bias','gradWeight','gradBias')
    end
    model_p:add(l_model):add(r_model)
    model:add(model_p)
    model:add(nn.CAddTable():cuda())
    print(model)

    local model_param, model_grad_param = model:getParameters()
    print(string.format('number of parameters: %d', model_param:nElement()))
    
    print(c.blue '==>' ..' loading parameters')
    -- load parameters
    local params = torch.load(model_params)
    assert(params:nElement() == model_param:nElement(), string.format('%s: %d vs %d', 'loading parameters: dimension mismatch.', params:nElement(), model_param:nElement()))
    model_param:copy(params)

    if(string.len(model_bnmeanstd) > 0) then 
        local bn_mean, bn_std = table.unpack(torch.load(model_bnmeanstd))
        print(bn_mean[k])

        for k,v in pairs(model:findModules('nn.SpatialBatchNormalization')) do
            v.running_mean:copy(bn_mean[k])
            v.running_var:copy(bn_std[k])
        end
    end
end

local function load_model()
    require('models/' .. model_name .. '.lua')
    model = create_model(half_range*2 + 1, 3):cuda()

    local model_param, model_grad_param = model:getParameters()
    print(string.format('number of parameters: %d', model_param:nElement()))
    
    print(c.blue '==>' ..' loading parameters')
    -- load parameters
    local params = torch.load(model_params)
    assert(params:nElement() == model_param:nElement(), string.format('%s: %d vs %d', 'loading parameters: dimension mismatch.', params:nElement(), model_param:nElement()))
    model_param:copy(params)

    if(string.len(model_bnmeanstd) > 0) then 
        local bn_mean, bn_std = table.unpack(torch.load(model_bnmeanstd))

        for k,v in pairs(model:findModules('nn.SpatialBatchNormalization')) do
            v.running_mean:copy(bn_mean[k])
            v.running_var:copy(bn_std[k])
        end
    end
end
     
acc_count = 0
function evaluate()
    -- compute 3-pixel error
    local l, r, tar, ll, rr = my_dataset:get_test_cuda()
    local n = (#l)[1]
    left_model = model
    left_model:evaluate()

    local remainder = math.fmod(n, opt.tb)
    print(remainder)
    if remainder ~= 0 then
        n = n - remainder
    end
    print(n)
    assert(math.fmod(n, opt.tb) == 0, "use opt.tb to be divided exactly by number of validate sample")
    acc_count = 0
    for i=1,n,opt.tb do
        o = left_model:forward({{l:narrow(1,i,opt.tb), r:narrow(1,i,opt.tb)}, {ll:narrow(1,i,opt.tb), rr:narrow(1,i,opt.tb)}})
        local _,y = o:max(2)
        y = y:long():cuda()
        acc_count = acc_count + (torch.abs(y-tar:narrow(1,i,opt.tb)):le(3):sum())
    end
    acc_count = acc_count / n * opt.tb
    print('Test accuracy: ' .. c.cyan(acc_count) .. ' %')

end

if opt.dual_net == 1 then
    load_model_dual()
else
    load_model()
end
evaluate()