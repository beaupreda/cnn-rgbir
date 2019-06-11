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

local TestDataHandler = torch.class('TestDataHandler')

function TestDataHandler:__init(data_root, testing, img_nb, psz, half_range, fold, gpu, offset)
    self.fold = fold
    self.channels = 3
    self.psz = psz
    self.pSize = 2 * psz + 1
    self.half_range = half_range
    self.cuda = gpu or 1

    local name = string.format('%s', testing)
    local file = io.open(name, 'r')
    local size = file:seek('end')
    -- every entry in binary file is 4 bytes
    size = size / 4
    self.locations = torch.FloatTensor(torch.FloatStorage(name, false, size)):view(-1,5)

    self.rgb = {}
    self.lwir = {}

    self.permutations = torch.randperm((#self.locations)[1])

    nb_points = (#self.locations)[1]
    print(string.format('#testing locations: %d ', (#self.locations)[1]))

    -- load and normalize test images
    for i = offset, img_nb + offset - 1 do
        j = i - offset
        xlua.progress(j + 1, img_nb)
        local rgb = image.load(string.format('%s/test/rgb/%d.png', data_root, i), self.channels, 'byte'):float()
        local lwir = image.load(string.format('%s/test/lwir/%d.png', data_root, i), self.channels, 'byte'):float()
        rgb:add(-rgb:mean()):div(rgb:std())
        lwir:add(-lwir:mean()):div(lwir:std())
        self.rgb[j] = rgb
        self.lwir[j] = lwir
    end
    
    self.left_rgb = torch.Tensor(nb_points, self.channels, self.pSize, self.pSize)
    self.left_lwir = torch.Tensor(nb_points, self.channels, self.pSize, self.pSize + self.half_range * 2)
    self.right_lwir = torch.Tensor(nb_points, self.channels, self.pSize, self.pSize)
    self.right_rgb = torch.Tensor(nb_points, self.channels, self.pSize, self.pSize + self.half_range * 2)
    self.labels = torch.Tensor(nb_points, 1):fill(self.half_range+1)

    for i = 1, nb_points do
        -- get center of patches for left network
        local id = self.locations[self.permutations[i]][1]
        local type = self.locations[self.permutations[i]][2]
        local x = self.locations[self.permutations[i]][3]
        local y = self.locations[self.permutations[i]][4]
        local right_x = self.locations[self.permutations[i]][5]
        
        -- small patch rgb
        self.left_rgb[i] = self.rgb[id - offset][{{}, {y - self.psz, y + self.psz}, {x - self.psz, x + self.psz}}]
        -- big patch lwir
        self.left_lwir[i] = self.lwir[id - offset][{{}, {y - self.psz, y + self.psz}, {right_x - self.psz - self.half_range, right_x + self.psz + self.half_range}}]

        -- swap x coordinates rgb-lwir for right network
        local tmp = x
        x = right_x
        right_x = tmp

        -- small patch lwir
        self.right_lwir[i] = self.lwir[id - offset][{{}, {y - self.psz, y + self.psz}, {x - self.psz, x + self.psz}}]
        -- big patch rgb
        self.right_rgb[i] = self.rgb[id - offset][{{}, {y - self.psz, y + self.psz}, {right_x - self.psz - self.half_range, right_x + self.psz + self.half_range}}]
    end
    collectgarbage()
end

function TestDataHandler:get_test()
    return self.left_rgb:cuda(), self.left_lwir:cuda(), self.right_lwir:cuda(), self.right_rgb:cuda(), self.labels:cuda()
end

