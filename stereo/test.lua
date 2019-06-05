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
    --folder_name              (default '')                    folder of .t7 files
    -g, --gpuid                (default 0)                     gpu id
    --test_num                 (default 10)                    number of test images
    --data_root                (default '')                    dataset root folder (images)
    --util_root                (default '')                    points location root folder (.bin files)
    --tb                       (default 100)                   test batch size
    --psz                      (default 18)                    half width
    --half_range               (default 60)                    half range
    --fold                     (default 1)                     fold number (1, 2 or 3)
]]

model_params = opt.folder_name .. '/param_epoch_200.t7'
model_bnmeanstd = opt.folder_name .. '/bn_meanvar_epoch_200.t7'

print(c.blue '==>' ..' configuring model')

torch.manualSeed(123)
gpu = opt.gpuid + 1
cutorch.setDevice(gpu)
torch.setdefaulttensortype('torch.FloatTensor')

-- number of the 1st image in the test set, depending on the fold
offset = -1
if opt.fold == 1 then
    offset = 1012
elseif opt.fold == 2 then
    offset = 940
elseif opt.fold == 3 then
    offset = 984

print(c.blue '==>' ..' loading test data')

dataset = TestDataHandler(opt.data_root, opt.util_root, opt.test_num, opt.psz, opt.half_range, opt.fold, gpu, offset)

local function load_model()
    require('model.lua')
    model = nn.Sequential()
    parallel = nn.ParallelTable()
    left = create_model(opt.half_range * 2 + 1, 3):cuda()
    right = create_model(opt.half_range * 2 + 1, 3):cuda()
    right = l_model:clone('weight','bias','gradWeight','gradBias')
    parallel:add(left):add(right)
    model:add(parallel)
    model:add(nn.CAddTable():cuda())

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
     
acc_count = 0
function evaluate()
    -- compute 3-pixel error
    local l, r, tar, ll, rr = dataset:get_test_cuda()
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

load_model()
evaluate()