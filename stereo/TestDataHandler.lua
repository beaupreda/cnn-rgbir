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

function TestDataHandler:__init(data_root, util_root, num_test_img, psz, half_range, gpu, test_name, extension, type, offset, invert)
    self.channels = 3
    self.invert = invert
    self.psz = psz
    self.pSize = 2*psz + 1
    self.half_range = half_range
    self.cuda = gpu or 1

    local full_test_name = string.format('%s/%s_%d_%d_%s.bin', util_root, test_name, self.psz, self.half_range, type)
    print(full_test_name)
    local test_file = io.open(full_test_name, 'r')
    local test_file_size = test_file:seek('end')
    test_file_size = test_file_size / 4
    self.test_loc = torch.FloatTensor(torch.FloatStorage(full_test_name, false, test_file_size)):view(-1,5)

    self.ldata = {}
    self.rdata = {}

    self.test_perm = torch.randperm((#self.test_loc)[1])

    num_test = (#self.test_loc)[1]
    print(string.format('#testing locations: %d ', (#self.test_loc)[1]))

    for i = offset, num_test_img + offset - 1 do
        j = i - offset
        xlua.progress(j, num_test_img)
        local l_img, r_img
        l_img = image.load(string.format('%s/test/rgb/%d%s', data_root, i, extension), self.channels, 'byte'):float()
        r_img = image.load(string.format('%s/test/lwir/%d%s', data_root, i, extension), self.channels, 'byte'):float()
        l_img:add(-l_img:mean()):div(l_img:std())
        r_img:add(-r_img:mean()):div(r_img:std())
        self.ldata[j] = l_img
        self.rdata[j] = r_img
    end
    
    self.test_left_rgb = torch.Tensor(num_test, self.channels, self.pSize, self.pSize)
    self.test_right_rgb = torch.Tensor(num_test, self.channels, self.pSize, self.pSize+self.half_range*2)
    self.test_left_lwir = torch.Tensor(num_test, self.channels, self.pSize, self.pSize)
    self.test_right_lwir = torch.Tensor(num_test, self.channels, self.pSize, self.pSize+self.half_range*2)
    self.test_label = torch.Tensor(num_test, 1):fill(self.half_range+1)

    for i = 1, num_test do
        local img_id, loc_type, center_x, center_y, right_center_x
        img_id, loc_type, center_x, center_y, right_center_x = self.test_loc[self.test_perm[i]][1], self.test_loc[self.test_perm[i]][2], self.test_loc[self.test_perm[i]][3], self.test_loc[self.test_perm[i]][4], self.test_loc[self.test_perm[i]][5]
        
        local right_center_y = center_y
        self.test_left_rgb[i] = self.ldata[img_id-offset][{{}, {center_y-self.psz, center_y+self.psz}, {center_x-self.psz, center_x+self.psz}}]
        self.test_right_rgb[i] = self.rdata[img_id-offset][{{}, {right_center_y-self.psz, right_center_y+self.psz}, {right_center_x-self.psz-self.half_range, right_center_x+self.psz+self.half_range}}]
        img_id, loc_type, right_center_x, center_y, center_x = self.test_loc[self.test_perm[i]][1], self.test_loc[self.test_perm[i]][2], self.test_loc[self.test_perm[i]][3], self.test_loc[self.test_perm[i]][4], self.test_loc[self.test_perm[i]][5]
        right_center_y = center_y
        self.test_left_lwir[i] = self.rdata[img_id-offset][{{}, {center_y-self.psz, center_y+self.psz}, {center_x-self.psz, center_x+self.psz}}]
        self.test_right_lwir[i] = self.ldata[img_id-offset][{{}, {right_center_y-self.psz, right_center_y+self.psz}, {right_center_x-self.psz-self.half_range, right_center_x+self.psz+self.half_range}}]
    end
    collectgarbage()
end

function TestDataHandler:get_test_cuda()
    return self.test_left_rgb:cuda(), self.test_right_rgb:cuda(), self.test_label:cuda(), self.test_left_lwir:cuda(), self.test_right_lwir:cuda()
end

