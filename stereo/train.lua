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
require 'TrainDataHandler'
local c = require 'trepl.colorize'
lapp = require 'pl.lapp'
opt = lapp[[
    --data_version             (default '')
    -s,--save                  (default 'logs/debug')       directory to save logs
    -b,--batchSize             (default 128)                batch size
    -g, --gpuid                (default 0)                  gpu id
    --train_nb                 (default 10)                 training images
    --validation_nb            (default 1)                  validation images
    -r,--learningRate          (default 1e-2)               learning rate
    --learningRateDecay        (default 1e-7)               learning rate decay
    --weightDecay              (default 0.0005)             weightDecay
    -m,--momentum              (default 0.9)                momentum
    --model                    (default dot_win37_dep9)     model name
    --epoch_step               (default 40)                 half learning rate at every 'epoch_step'
    --weight_epoch             (default 5)                  save weight at every 'weight_epoch'
    --max_epoch                (default 10)                 maximum number of iterations
    --iter_per_epoch           (default 50)                 evaluate every # iterations, and update plot
    --data_root                (default '')                 dataset root folder
    --util_root                (default '')                 dataset root folder
    --tb                       (default 100)                validation batch size
    --validation_points        (default 10000)              number validation patch pair
    --opt_method               (default 'adam')             sgd, adagrad, adam
    --showCurve                (default 0)                  use 1 to show training / validation curve
    --psz                      (default 9)                  half width
    --half_range               (default 100)                half range
]]

print(c.blue '==>' ..' configuring model')

torch.manualSeed(42)
local gpu = cutorch.setDevice(opt.gpuid+1)
torch.setdefaulttensortype('torch.FloatTensor')

print(c.blue '==>' ..' loading data')
dataset = TrainDataHandler(opt.data_root, opt.util_root, opt.train_nb, opt.validation_nb, opt.validation_points, opt.batchSize, opt.psz, opt.half_range, gpu)

-- model creation
require('model.lua')
left = create_model(opt.half_range * 2 + 1, 3):cuda()
right = left:clone('weight', 'bias', 'gradWeight', 'gradBias')

model = nn.Sequential()
parallel = nn.ParallelTable()
parallel:add(left):add(right)
model:add(parallel)
model:add(nn.CAddTable():cuda())
parameters, gradParameters = model:getParameters()
print(string.format('number of parameters: %d', parameters:nElement()))

local folder_name = 'logs/' .. get_folder_name()
paths.mkdir(folder_name)
testLogger = optim.Logger(paths.concat(folder_name, 'validation.log'))
argsFile = io.open(paths.concat(folder_name, 'args.log'), 'w')
save_arguments()
testLogger:setNames{'% mean patch pred accuracy (train set)', '% mean patch pred accuracy (validation set)','% loss (train set)', '% loss (validation set)'}
if opt.showCurve == 0 then
    testLogger.showPlot = false
end

print(c.blue'==>' ..' setting criterion')
-- smooth target distribution
gt_weight = torch.Tensor({2, 5, 10, 16, 10, 5, 2})
criterion = nn.MulClassNLLCriterion(gt_weight):cuda()

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
    model:training()
    local tic = torch.tic()
    train_loss = -1
    local good_predictions = 0
    for t = 1, opt.iter_per_epoch do
        xlua.progress(t, opt.iter_per_epoch)
        local left_rgb, left_lwir, right_lwir, right_rgb, target = dataset:next_batch()
        local feval = function(x)
            if x ~= parameters then parameters:copy(x) end
            gradParameters:zero()
            local output = model:forward({{left_rgb, left_lwir}, {right_lwir, right_rgb}})
            local f = criterion:forward(output, target)
            local gradient = criterion:backward(output, target)
            model:backward({{left_rgb, left_lwir}, {right_rgb, right_lwir}}, gradient)
            local _, y = output:max(2)
            y = y:long():cuda()
            -- 3 pixel error
            good_predictions = good_predictions + (torch.abs(y - target):le(3):sum())
            return f, gradParameters
        end

        local _, loss = optimMethod(feval, parameters, optimConfig, optimState)
        train_loss = loss[1]
    end

    accuracy = good_predictions / (opt.iter_per_epoch * opt.batchSize) * 100
    print(('Train loss: ' .. c.cyan '%.4f' .. ' train accuracy: ' .. c.cyan '%.2f' .. ' %%\t time: %.2f s\t grad/param norm = %6.4e\t learning rate: %f'):format(train_loss / opt.batchSize, accuracy, torch.toc(tic), gradParameters:norm() / parameters:norm(), optimConfig.learningRate))
    epoch = epoch + 1
end

acc_count = 0
function evaluate()
    model:evaluate()
    local left_rgb, left_lwir, right_lwir, right_rgb, target = dataset:get_eval_cuda()
    local nb_points = (#left_rgb)[1]
    loss = -1

    assert(math.fmod(n, opt.tb) == 0, "use opt.tb to be divided exactly by number of validate sample")
    local good_predictions = 0
    for i = 1, nb_points, opt.tb do
        output = model:forward({{left_rgb:narrow(1, i, opt.tb), left_lwir:narrow(1, i, opt.tb)}, {right_lwir:narrow(1, i, opt.tb), right_rgb:narrow(1, i, opt.tb)}})
        local f = criterion:forward(output, target)
        local _, y = o:max(2)
        y = y:long():cuda()
        -- 3 pixel error
        good_predictions = good_predictions + (torch.abs(y - target:narrow(1, i, opt.tb)):le(3):sum())
        loss = f
    end

    validation_loss = loss / n * 100
    good_predictions = good_predictions / n * opt.tb
    print('Test accuracy: ' .. c.cyan(good_predictions) .. ' %')
    print('Validation loss: ' .. c.cyan(validation_loss))

    if epoch % 10 == 0 then collectgarbage() end
end

function logging( )
    if testLogger then
        paths.mkdir(folder_name)
        testLogger:add{accuracy, good_predictions, train_loss / opt.batchSize, validation_loss}
        testLogger:style{'-', '-', '-', '-'}
        testLogger:plot()
        os.execute('convert -density 200 ' .. folder_name .. '/test.log.eps ' .. folder_name .. '/test.png')
    end

    -- save model parameters every # epochs
    if epoch % opt.weight_epoch == 0 or epoch == opt.max_epoch then
        local filename = paths.concat(folder_name, string.format('param_epoch_%d.t7', epoch))
        print('==> saving parameters to ' .. filename)
        torch.save(filename, parameters)

        -- save bn statistics from training set
        filename = paths.concat(folder_name, string.format('bn_meanvar_epoch_%d.t7', epoch))
        print('==> saving bn mean var to ' .. filename)
        local bn_mean = {}
        local bn_var = {}
        for k, v in pairs(model:findModules('nn.SpatialBatchNormalization')) do
            bn_mean[k] = v.running_mean
            bn_var[k] = v.running_var
        end

        if #bn_mean > 0 then torch.save(filename, {bn_mean, bn_var}) end
    end
end

-- each run is saved in a folder identified by the start time of the execution
local function get_folder_name()
    local date_table = os.date('*t')
    local year, month, day = date_table.year, date_table.month, date_table.day
    local hour, minute, second = date_table.hour, date_table.min, date_table.sec
    local result = string.format('run_%dy_%dm_%dd_%dh_%dmin_%dsec', year, month, day, hour, minute, second)
    return result
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
    argsFile:write(string.format('--validation_points                number of test patch pair                   %d\n', opt.validation_points))
    argsFile:write(string.format('--opt_method                 optimization method (sgd, adam, adagrad)          %s\n', opt.opt_method))
    argsFile:write(string.format('--showCurve                  show or not accuracy curves                       %d\n', opt.showCurve))
    argsFile:write(string.format('--psz                        half width                                        %d\n', opt.psz))
    argsFile:write(string.format('--half_range                 half range                                        %d\n', opt.half_range))
    argsFile:write(string.format('--model_param                loaded model parameters for finetuning            %s\n', opt.model_param))
    argsFile:write(string.format('--bn_meanstd                 loaded batch norm mean and standard dev           %s\n', opt.bn_meanstd))
end

while epoch < opt.max_epoch do
    train()
    if opt.validation_points > 0 then
        evaluate()
    end

    logging()
end


