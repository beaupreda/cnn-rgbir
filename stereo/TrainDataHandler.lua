-- data handler for RGB-IR images
--
-- David-Alexandre Beaupre
--

require 'image'
require 'cutorch'
require 'gnuplot'
require 'xlua'
require 'math'
require 'io'

local TrainDataHandler = torch.class('TrainDataHandler')

function TrainDataHandler:__init(data_root, util_root, train_nb, validation_nb, validation_points, batch_size, psz, half_range, fold, gpu)
    self.fold = fold
    self.channels = 3
    self.batch_size = batch_size
    self.psz = psz
    self.pSize = 2 * psz + 1
    self.half_range = half_range
    self.cuda = gpu or 1
    self.training_ptr = 0
    self.curr_epoch = 0

    local name = string.format('%s/train%d.bin', util_root, self.fold)
    local file = io.open(name, 'r')
    local size = file:seek('end')
    size = size / 4
    self.train_locations = torch.FloatTensor(torch.FloatStorage(name, false, size)):view(-1,5)
    self.train_permutations = torch.randperm((#self.train_locations)[1])

    name = string.format('%s/validation%d.bin', util_root, self.fold)
    file = io.open(name, 'r')
    size = file:seek('end')
    size = size / 4
    self.validation_locations = torch.FloatTensor(torch.FloatStorage(name, false, size)):view(-1,5)
    self.validation_permutations = torch.randperm((#self.validation_locations)[1])

    self.rgb = {}
    self.lwir = {}

    print(string.format('#training locations: %d -- #validation locations: %d', (#self.train_locations)[1], (#self.validation_permutations)[1]))

    local i = 0
    for j = i, train_nb do
        xlua.progress(j, train_nb)
        local rgb = image.load(string.format('%s/train/rgb/%d.png', data_root, j), self.channels, 'byte'):float()
        local lwir = image.load(string.format('%s/train/lwir/%d.png', data_root, j), self.channels, 'byte'):float()
        rgb:add(-rgb:mean()):div(rgb:std())
        lwir:add(-lwir:mean()):div(lwir:std())
        self.rgb[j] = rgb
        self.lwir[j] = lwir
    end

    i = i + train_nb + 1
    for j = i, validation_nb do
        xlua.progress(j, validation_nb)
        local rgb = image.load(string.format('%s/validation/rgb/%d.png', data_root, j), self.channels, 'byte'):float()
        local lwir = image.load(string.format('%s/validation/lwir/%d.png', data_root, j), self.channels, 'byte'):float()
        rgb:add(-rgb:mean()):div(rgb:std())
        lwir:add(-lwir:mean()):div(lwir:std())
        self.rgb[j] = rgb
        self.lwir[j] = lwir
    end

    print(string.format('receptive field size: %d; total range: %d', self.pSize, self.half_range*2+1))

    self.bleft_rgb = torch.Tensor(self.batch_size, self.channels_rgb, self.pSize, self.pSize)
    self.bleft_lwir = torch.Tensor(self.batch_size, self.channels_ir, self.pSize, self.pSize + self.half_range * 2)
    self.bright_lwir = torch.Tensor(self.batch_size, self.channels_rgb, self.pSize, self.pSize)
    self.bright_rgb = torch.Tensor(self.batch_size, self.channels_ir, self.pSize, self.pSize + self.half_range * 2)
    self.blabels = torch.Tensor(self.batch_size, 1):fill(self.half_range + 1)

    self.vleft_rgb = torch.Tensor(validation_points, self.channels_rgb, self.pSize, self.pSize)
    self.vleft_lwir = torch.Tensor(validation_points, self.channels_ir, self.pSize, self.pSize + self.half_range * 2)
    self.vright_lwir = torch.Tensor(validation_points, self.channels_rgb, self.pSize, self.pSize)
    self.vright_rgb = torch.Tensor(validation_points, self.channels_ir, self.pSize, self.pSize + self.half_range * 2)
    self.vlabels = torch.Tensor(validation_points, 1):fill(self.half_range + 1)

    for i = 1, validation_points do
        -- get center of patches for left network
        local id = self.validation_locations[self.validation_permutations[i]][1]
        local type = self.validation_locations[self.validation_permutations[i]][2]
        local x = self.validation_locations[self.validation_permutations[i]][3]
        local y = self.validation_locations[self.validation_permutations[i]][4]
        local right_x = self.validation_locations[self.validation_permutations[i]][5]

        -- small patch rgb
        self.vleft_rgb[i] = self.rgb[id][{{}, {y - self.psz, y + self.psz}, {x - self.psz, x + self.psz}}]
        -- big patch lwir
        self.vleft_lwir[i] = self.lwir[id][{{}, {y - self.psz, y + self.psz}, {right_x - self.psz - self.half_range, right_x + self.psz + self.half_range}}]

        -- swap x coordinates for right network
        local tmp = x
        x = right_x
        right_x = tmp        
    
        -- small patch lwir
        self.vright_lwir[i] = self.lwir[id][{{}, {y - self.psz, y + self.psz}, {x - self.psz, x + self.psz}}]
        -- big patch rgb
        self.vright_rgb[i] = self.rgb[id][{{}, {y - self.psz, y + self.psz}, {right_x - self.psz - self.half_range, right_x + self.psz + self.half_range}}]
    end

    collectgarbage()
end

function TrainDataHandler:next_batch()
    for idx = 1, self.batch_size do
        local i = self.training_ptr + idx
        if i > torch.numel(self.tr_perm) then
            i = 1
            self.training_ptr = -idx + 1
        end

        -- get center of patches for left network
        local id = self.train_locations[self.train_permutations[i]][1]
        local type = self.train_locations[self.train_permutations[i]][2]
        local x = self.train_locations[self.train_permutations[i]][3]
        local y = self.train_locations[self.train_permutations[i]][4]
        local right_x = self.train_locations[self.train_permutations[i]][5]

        -- small patch rgb
        self.bleft_rgb[idx] = self.rgb[id][{{}, {y - self.psz, y + self.psz}, {x - self.psz, x + self.psz}}]
        -- big patch lwir
        self.bleft_lwir[idx] = self.lwir[id][{{}, {y - self.psz, y + self.psz}, {right_x - self.psz - self.half_range, right_x + self.psz + self.half_range}}]

        -- swap x coordinates for right network
        local tmp = x
        x = right_x
        right_x = tmp

        -- small patch lwir
        self.bright_lwir[idx] = self.lwir[id][{{}, {y - self.psz, y + self.psz}, {x - self.psz, x + self.psz}}]
        -- big patch rgb
        self.bright_rgb[idx] = self.rgb[id][{{}, {y - self.psz, y + self.psz}, {right_x - self.psz - self.half_range, right_x + self.psz + self.half_range}}]
    end

    self.training_ptr = self.training_ptr + self.batch_size
    
    return self.bleft_rgb:cuda(), self.bleft_lwir:cuda(), self.bright_lwir:cuda(), self.bright_rgb:cuda(), self.blabels:cuda()
end 

function TrainDataHandler:get_eval()
    return self.vleft_rgb:cuda(), self.vleft_lwir:cuda(), self.vright_lwir:cuda(), self.vright_rgb:cuda(), self.vlabels:cuda()
end

