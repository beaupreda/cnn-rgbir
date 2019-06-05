-- train with 3 pixel weighted log-loss 
--
-- Wenjie Luo
-- David-Alexandre Beaupre
--

require 'xlua'
require 'optim'
require 'cunn'
require 'gnuplot'

require 'MulClassNLLCriterion'
require 'DataHandler'
local c = require 'trepl.colorize'
lapp = require 'pl.lapp'
opt = lapp[[
    --data_version             (default "kitti2015")
    -s,--save                  (default "logs/debug")       directory to save logs
    -b,--batchSize             (default 128)                batch size
    -g, --gpuid                (default 0)                  gpu id
    --tr_num                   (default 10)                 training images
    --val_num                  (default 1)                  validation images
    -r,--learningRate          (default 1e-2)               learning rate
    --learningRateDecay        (default 1e-7)               learning rate decay
    --weightDecay              (default 0.0005)             weightDecay
    -m,--momentum              (default 0.9)                momentum
    --model                    (default dot_win37_dep9)     model name
    --epoch_step               (default 40)                 half learning rate at every 'epoch_step'
    --weight_epoch             (default 5)                  save weight at every 'weight_epoch'
    --max_epoch                (default 10)                 maximum number of iterations
    --iter_per_epoch           (default 50)                 evaluate every # iterations, and update plot
    --data_root                (default "/ais/gobi3/datasets/kitti/scene_flow/training") dataset root folder
    --util_root                (default "")                 dataset root folder
    --tb                       (default 100)                test batch size
    --num_val_loc              (default 10000)              number test patch pair
    --opt_method               (default 'adam')             sgd, adagrad, adam
    --type                     (default 'normal')           normal, small, tiny (size of dataset used)
    --invert                   (default 0)                  switch lwir-rgb images
    --full_eval                (default 0)                  evaluate with both models or the left one
    --share                    (default 0)                  share params or not

    --showCurve                (default 0)                  use 1 to show training / validation curve
    --psz                      (default 9)                  half width
    --half_range               (default 100)                half range
    --fine_tune                (default 0)                  finetune model with new training data
    --model_param              (default '')                 weight file
    --bn_meanstd               (default '')
]]

local function get_folder_name()
    local date_table = os.date('*t')
    local hour, minute, second = date_table.hour, date_table.min, date_table.sec
    local year, month, day = date_table.year, date_table.month, date_table.day
    local result = string.format('run_%dy_%dm_%dd_%dh_%dmin_%dsec', year, month, day, hour, minute, second)
    return result
end

local function load_model()
    require('models/' .. opt.model .. '.lua')
    model = create_model(opt.half_range*2 + 1, 3):cuda()

    local model_param, model_grad_param = model:getParameters()
    print(string.format('number of parameters: %d', model_param:nElement()))
    
    print(c.blue '==>' ..' loading parameters')
    -- load parameters
    local params = torch.load(opt.model_param)
    assert(params:nElement() == model_param:nElement(), string.format('%s: %d vs %d', 'loading parameters: dimension mismatch.', params:nElement(), model_param:nElement()))
    model_param:copy(params)

    if(string.len(opt.bn_meanstd) > 0) then 
        local bn_mean, bn_std = table.unpack(torch.load(opt.bn_meanstd))

        for k,v in pairs(model:findModules('nn.SpatialBatchNormalization')) do
            v.running_mean:copy(bn_mean[k])
            v.running_var:copy(bn_std[k])
        end
        model:evaluate()
    end
end

function save_arguments()
    argsFile:write(string.format('Parameter                    Definition                                        Value\n'))
    argsFile:write(string.format('--data_version               dataset_version                                   %s\n', opt.data_version))
    argsFile:write(string.format('--data_root                  images location                                   %s\n', opt.data_root))
    argsFile:write(string.format('--util_root                  binary files location                             %s\n', opt.util_root))
    argsFile:write(string.format('--batchSize                  batch size                                        %d\n', opt.batchSize))
    argsFile:write(string.format('--tr_num                     number of training images                         %d\n', opt.tr_num))
    argsFile:write(string.format('--val_num                    number of validation images                       %d\n', opt.val_num))
    argsFile:write(string.format('--learningRate               learning rate                                     %f\n', opt.learningRate))
    argsFile:write(string.format('--learningRateDecay          decay of learning rate                            %f\n', opt.learningRateDecay))
    argsFile:write(string.format('--weightDecay                weight decay                                      %f\n', opt.weightDecay))
    argsFile:write(string.format('--momentum                   momentum                                          %f\n', opt.momentum))
    argsFile:write(string.format('--model                      which model is used                               %s\n', opt.model))
    argsFile:write(string.format('--epoch_step                 half learning rate at every epoch step            %d\n', opt.epoch_step))
    argsFile:write(string.format('--weight_epoch               save weight at every weight epoch                 %d\n', opt.weight_epoch))
    argsFile:write(string.format('--max_epoch                  maximum number of iterations                      %d\n', opt.max_epoch))
    argsFile:write(string.format('--iter_per_epoch             evaluate every # of iterations                    %d\n', opt.iter_per_epoch))
    argsFile:write(string.format('--tb                         test batch size                                   %d\n', opt.tb))
    argsFile:write(string.format('--num_val_loc                number of test patch pair                         %d\n', opt.num_val_loc))
    argsFile:write(string.format('--opt_method                 optimization method (sgd, adam, adagrad)          %s\n', opt.opt_method))
    argsFile:write(string.format('--type                       size of the dataset used                          %s\n', opt.type))
    argsFile:write(string.format('--invert                     switch rgb-lwir                                   %d\n', opt.invert))
    argsFile:write(string.format('--full_eval                  eval with both models or left one                 %d\n', opt.full_eval))
    argsFile:write(string.format('--share                      share params or not                               %d\n', opt.share))
    argsFile:write(string.format('--showCurve                  show or not accuracy curves                       %d\n', opt.showCurve))
    argsFile:write(string.format('--psz                        half width                                        %d\n', opt.psz))
    argsFile:write(string.format('--half_range                 half range                                        %d\n', opt.half_range))
    argsFile:write(string.format('--fine_tune                  finetuning model with new training data or not    %d\n', opt.fine_tune))
    argsFile:write(string.format('--model_param                loaded model parameters for finetuning            %s\n', opt.model_param))
    argsFile:write(string.format('--bn_meanstd                 loaded batch norm mean and standard dev           %s\n', opt.bn_meanstd))
end

print(opt)

print(c.blue '==>' ..' configuring model')

torch.manualSeed(123)
cutorch.setDevice(opt.gpuid+1)
torch.setdefaulttensortype('torch.FloatTensor')

extension = '.jpg'
val_bin = 'gab_val'
tr_bin = 'gab_train'
if opt.data_version == 'stcharles' then
    extension = '.png'
    val_bin = 'plsc_val'
    tr_bin = 'plsc_train'
elseif opt.data_version == 'all' then
    extension = '.png'
    val_bin = 'all_val'
    tr_bin = 'all_train'
end

if opt.invert == 1 then
    val_bin = val_bin .. '_inv'
    tr_bin = tr_bin .. '_inv'
end

data_type = opt.type

print(c.blue '==>' ..' loading data')
my_dataset = DataHandlerTwoNetworks(opt.data_root, opt.util_root, opt.tr_num, opt.val_num, opt.num_val_loc, opt.batchSize, opt.psz, opt.half_range, 1, val_bin, tr_bin, extension, data_type)

if (opt.fine_tune == 0) then 
    print(c.blue '==>' ..' create model')
    require('models/' .. opt.model .. '.lua')
    if opt.data_version == 'kitti2015' or opt.data_version == 'bilodeau' or opt.data_version == 'stcharles' or opt.data_version == 'all' then
        model_rgb_lwir = create_model(opt.half_range*2 + 1, 3):cuda()
        model_lwir_rgb = model_rgb_lwir:clone('weight','bias','gradWeight','gradBias')
        if opt.share == 1 then
            model_lwir_rgb = create_model(opt.half_range*2 + 1, 3):cuda()
        end
    elseif opt.data_version == 'kitti2012' then
        model = create_model(opt.half_range*2 + 1, 1):cuda()
    else
        error('data_version should be either kitti2012 or kitti2015')
    end
else
    load_model()
    parameters,gradParameters = model:getParameters()
    print('FINETUNE\n')
end

model = nn.Sequential()
model_p = nn.ParallelTable()
model_p:add(model_rgb_lwir):add(model_lwir_rgb)
model:add(model_p)
model:add(nn.CAddTable():cuda())
print(model)
parameters,gradParameters = model:getParameters()
print(string.format('number of parameters: %d', parameters:nElement()))

folder_name = 'logs/' .. get_folder_name()
print('Will save at ' .. folder_name)
paths.mkdir(folder_name)
testLogger = optim.Logger(paths.concat(folder_name, 'validation.log'))
argsFile = io.open(paths.concat(folder_name, 'args.log'), 'w')
save_arguments()
testLogger:setNames{'% mean patch pred accuracy (train set)', '% mean patch pred accuracy (validation set)','% loss (train set)', '% loss (validation set)'}
if opt.showCurve == 0 then
    testLogger.showPlot = false
end

print(c.blue'==>' ..' setting criterion')
gt_weight = torch.Tensor({2,5,10,16,10,5,2})--torch.ones(7)
criterion_rgb_lwir = nn.MulClassNLLCriterion(gt_weight):cuda()
criterion_lwir_rgb = nn.MulClassNLLCriterion(gt_weight):cuda()
parallel_criterion = nn.MulClassNLLCriterion(gt_weight):cuda()

print(c.blue'==>' ..' configuring optimizer')
optimConfig = {
    learningRate = opt.learningRate,
    weightDecay = opt.weightDecay,
    momentum = opt.momentum,
    learningRateDecay = opt.learningRateDecay,
}
optimState = {}
optimMethod = optim[opt.opt_method]
epoch = epoch or 1
function train()
    local acc_count = 0
    model:training()
    -- drop learning rate every "epoch_step" epochs
    if epoch == 120 then optimConfig.learningRate = optimConfig.learningRate/5 end
    if epoch > 120 and (epoch - 120) % opt.epoch_step == 0 then optimConfig.learningRate = optimConfig.learningRate/5 end

    print(c.blue '==>'.." fake epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']')

    local tic = torch.tic()
    train_loss = -1
    for t = 1, opt.iter_per_epoch do
        xlua.progress(t, opt.iter_per_epoch)
        
        local left_rgb_lwir, right_rgb_lwir, targets_rgb_lwir, left_lwir_rgb, right_lwir_rgb, targets_lwir_rgb = my_dataset:next_batch()
        local feval = function(x)
            if x ~= parameters then parameters:copy(x) end
            gradParameters:zero()
            local outputs = model:forward({{left_rgb_lwir, right_rgb_lwir}, {left_lwir_rgb, right_lwir_rgb}})
            local f = parallel_criterion:forward(outputs, targets_rgb_lwir)
            local df_do = parallel_criterion:backward(outputs, targets_rgb_lwir)
            model:backward({{left_rgb_lwir, right_rgb_lwir}, {left_lwir_rgb, right_lwir_rgb}}, df_do)

            -- 3 pixel error
            local _, y_rgb_lwir = outputs:max(2)
            y_rgb_lwir = y_rgb_lwir:long():cuda()
            acc_count = acc_count + (torch.abs(y_rgb_lwir-targets_rgb_lwir):le(3):sum())-- + 0.5*(torch.abs(y_lwir_rgb-targets_lwir_rgb):le(3):sum())

            return f,gradParameters
        end
        local _, loss = optimMethod(feval, parameters, optimConfig, optimState)
        train_loss = loss[1]
    end

    train_acc = acc_count/(opt.iter_per_epoch*opt.batchSize) * 100
    print(('Train loss: '..c.cyan'%.4f' .. ' train accuracy: '..c.cyan'%.2f'..' %%\t time: %.2f s\t grad/param norm = %6.4e\t learning rate: %f'):format(train_loss/opt.batchSize, train_acc, torch.toc(tic), gradParameters:norm() / parameters:norm(), optimConfig.learningRate))

    epoch = epoch + 1
end

acc_count = 0
function evaluate()
    -- compute 3-pixel error
    local l_rgb_lwir, r_rgb_lwir, tar_rgb_lwir, l_lwir_rgb, r_lwir_rgb, tar_lwir_rgb = my_dataset:get_eval_cuda()
    local n = (#l_rgb_lwir)[1]
    loss = -1
    if opt.full_eval == 0 then
        print('LEFT MODEL EVAL')
        model:evaluate()

        assert(math.fmod(n, opt.tb) == 0, "use opt.tb to be divided exactly by number of validate sample")
        acc_count = 0
        for i=1,n,opt.tb do
            o = model:forward({{l_rgb_lwir:narrow(1,i,opt.tb), r_rgb_lwir:narrow(1,i,opt.tb)}, {l_lwir_rgb:narrow(1,i,opt.tb), r_lwir_rgb:narrow(1,i,opt.tb)}})
            local f = criterion_rgb_lwir:forward(o, tar_rgb_lwir)
            local _,y_rl = o:max(2)
            y_rl = y_rl:long():cuda()
            acc_count = acc_count + (torch.abs(y_rl-tar_rgb_lwir:narrow(1,i,opt.tb)):le(3):sum())
            loss = f
        end
    else
        print('FULL MODEL EVAL')
        model:evaluate()
        print(c.blue '==>'.." validation")

        assert(math.fmod(n, opt.tb) == 0, "use opt.tb to be divided exactly by number of validate sample")
        acc_count = 0
        for i=1,n,opt.tb do
            o = model:forward({{l_rgb_lwir:narrow(1,i,opt.tb), r_rgb_lwir:narrow(1,i,opt.tb)}, {l_lwir_rgb:narrow(1,i,opt.tb), r_lwir_rgb:narrow(1,i,opt.tb)}})
            local f = parallel_criterion:forward(o, {tar_rgb_lwir, tar_lwir_rgb})
            o_rgb_lwir = o[1]
            o_lwir_rgb = o[2]
            local _,y_rl = o_rgb_lwir:max(2)
            local _,y_lr = o_lwir_rgb:max(2)
            y_rl = y_rl:long():cuda()
            y_lr = y_lr:long():cuda()
            acc_count = acc_count + 0.5*(torch.abs(y_rl-tar_rgb_lwir:narrow(1,i,opt.tb)):le(3):sum()) + 0.5*(torch.abs(y_lr-tar_lwir_rgb:narrow(1,i,opt.tb)):le(3):sum())
            loss = f
        end
    end
    val_loss = loss / n * 100
    acc_count = acc_count / n * opt.tb
    print('Test accuracy: ' .. c.cyan(acc_count) .. ' %')
    print('Validation loss: ' .. c.cyan(val_loss))

    if epoch % 10 == 0 then collectgarbage() end
end

function logging( )
    
    if testLogger then
        paths.mkdir(folder_name)
        testLogger:add{train_acc, acc_count, train_loss/opt.batchSize, val_loss}
        testLogger:style{'-','-','-','-'}
        testLogger:plot()
        
        os.execute('convert -density 200 '..folder_name..'/test.log.eps '..folder_name..'/test.png')
    end

    -- save model parameters every # epochs
    if epoch % opt.weight_epoch == 0 or epoch == opt.max_epoch then
        local filename = paths.concat(folder_name, string.format('param_epoch_%d.t7', epoch))
        print('==> saving parameters to '..filename)
        torch.save(filename, parameters)

        -- save bn statistics from training set
        filename = paths.concat(folder_name, string.format('bn_meanvar_epoch_%d.t7', epoch))
        print('==> saving bn mean var to '..filename)
        local bn_mean = {}
        local bn_var = {}
        for k,v in pairs(model:findModules('nn.SpatialBatchNormalization')) do
            bn_mean[k] = v.running_mean
            bn_var[k] = v.running_var
        end
        if #bn_mean > 0 then torch.save(filename, {bn_mean, bn_var}) end
    end
end

while epoch < opt.max_epoch do
    train()

    if opt.num_val_loc > 0 then
        evaluate()
    end

    logging()
end


