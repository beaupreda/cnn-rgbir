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

function TrainDataHandler:__init(data_root, util_root, num_tr_img, num_val_img, num_val_loc, batch_size, psz, half_range, gpu, val_name, tr_name, extension, type)
    self.channels_ir = 3
    self.channels_rgb = 3
    self.batch_size = batch_size
    self.psz = psz
    self.pSize = 2*psz + 1
    self.half_range = half_range
    self.cuda = gpu or 1
    self.tr_ptr = 0
    self.curr_epoch = 0

    local full_tr_name = string.format('%s/%s_%d_%d_%s.bin', util_root, tr_name, self.psz, self.half_range, type)
    local full_val_name = string.format('%s/%s_%d_%d_%s.bin', util_root, val_name, self.psz, self.half_range, type)
    print(full_tr_name)
    print(full_val_name)
    local tr_file = io.open(full_tr_name, 'r')
    local val_file = io.open(full_val_name, 'r')
    local tr_file_size = tr_file:seek('end')
    tr_file_size = tr_file_size / 4
    local val_file_size = val_file:seek('end')
    val_file_size = val_file_size / 4
    self.tr_loc = torch.FloatTensor(torch.FloatStorage(full_tr_name, false, tr_file_size)):view(-1,5)
    self.val_loc = torch.FloatTensor(torch.FloatStorage(full_val_name, false, val_file_size)):view(-1,5)

    self.ldata = {}
    self.rdata = {}
    self.lval = {}
    self.rval = {}
    self.data_root = data_root

    self.tr_perm = torch.randperm((#self.tr_loc)[1])
    self.val_perm = torch.randperm((#self.val_loc)[1])

    print(string.format('#training locations: %d -- #validation locations: %d', (#self.tr_loc)[1], (#self.val_loc)[1]))

    for i = 1, num_tr_img + num_val_img do
        xlua.progress(i, num_tr_img + num_val_img)
        local l_img_rgb_lwir, r_img_rgb_lwir
        if i <= num_tr_img then
            l_img_rgb_lwir = image.load(string.format('%s/train/rgb/%d%s', data_root, i - 1, extension), self.channels_rgb, 'byte'):float()
            r_img_rgb_lwir = image.load(string.format('%s/train/lwir/%d%s', data_root, i - 1, extension), self.channels_ir, 'byte'):float()
        else
            l_img_rgb_lwir = image.load(string.format('%s/validation/rgb/%d%s', data_root, i - 1, extension), self.channels_rgb, 'byte'):float()
            r_img_rgb_lwir = image.load(string.format('%s/validation/lwir/%d%s', data_root, i - 1, extension), self.channels_ir, 'byte'):float()
        end
        l_img_rgb_lwir:add(-l_img_rgb_lwir:mean()):div(l_img_rgb_lwir:std())
        r_img_rgb_lwir:add(-r_img_rgb_lwir:mean()):div(r_img_rgb_lwir:std())
        self.ldata[i - 1] = l_img_rgb_lwir
        self.rdata[i - 1] = r_img_rgb_lwir
    end

    print(string.format('receptive field size: %d; total range: %d', self.pSize, self.half_range*2+1))
    self.batch_left_rgb_lwir = torch.Tensor(self.batch_size, self.channels_rgb, self.pSize, self.pSize)
    self.batch_right_rgb_lwir = torch.Tensor(self.batch_size, self.channels_ir, self.pSize, self.pSize+self.half_range*2)
    self.batch_label = torch.Tensor(self.batch_size, 1):fill(self.half_range+1)
    self.batch_left_lwir_rgb = torch.Tensor(self.batch_size, self.channels_rgb, self.pSize, self.pSize)
    self.batch_right_lwir_rgb = torch.Tensor(self.batch_size, self.channels_ir, self.pSize, self.pSize+self.half_range*2)

    self.val_left_rgb_lwir = torch.Tensor(num_val_loc, self.channels_rgb, self.pSize, self.pSize)
    self.val_right_rgb_lwir = torch.Tensor(num_val_loc, self.channels_ir, self.pSize, self.pSize+self.half_range*2)
    self.val_label = torch.Tensor(num_val_loc, 1):fill(self.half_range+1)
    self.val_left_lwir_rgb = torch.Tensor(num_val_loc, self.channels_rgb, self.pSize, self.pSize)
    self.val_right_lwir_rgb = torch.Tensor(num_val_loc, self.channels_ir, self.pSize, self.pSize+self.half_range*2)

    for i = 1, num_val_loc do
        local img_id_rgb_lwir, loc_type_rgb_lwir, center_x_rgb_lwir, center_y_rgb_lwir, right_center_x_rgb_lwir, img_id_lwir_rgb, loc_type_lwir_rgb, center_x_lwir_rgb, center_y_lwir_rgb, right_center_x_lwir_rgb
        img_id_rgb_lwir, loc_type_rgb_lwir, center_x_rgb_lwir, center_y_rgb_lwir, right_center_x_rgb_lwir = self.val_loc[self.val_perm[i]][1], self.val_loc[self.val_perm[i]][2], self.val_loc[self.val_perm[i]][3], self.val_loc[self.val_perm[i]][4], self.val_loc[self.val_perm[i]][5]
        img_id_lwir_rgb, loc_type_lwir_rgb, right_center_x_lwir_rgb, center_y_lwir_rgb, center_x_lwir_rgb = self.val_loc[self.val_perm[i]][1], self.val_loc[self.val_perm[i]][2], self.val_loc[self.val_perm[i]][3], self.val_loc[self.val_perm[i]][4], self.val_loc[self.val_perm[i]][5]
        
        self.val_left_rgb_lwir[i] = self.lval[img_id_rgb_lwir - num_tr_img + 1][{{}, {center_y_rgb_lwir-self.psz, center_y_rgb_lwir+self.psz}, {center_x_rgb_lwir-self.psz, center_x_rgb_lwir+self.psz}}]
        self.val_right_rgb_lwir[i] = self.rval[img_id_rgb_lwir - num_tr_img + 1][{{}, {right_center_y_rgb_lwir-self.psz, right_center_y_rgb_lwir+self.psz}, {right_center_x_rgb_lwir-self.psz-self.half_range, right_center_x_rgb_lwir+self.psz+self.half_range}}]
        self.val_left_lwir_rgb[i] = self.lval[img_id_lwir_rgb - num_tr_img + 1][{{}, {center_y_lwir_rgb-self.psz, center_y_lwir_rgb+self.psz}, {center_x_lwir_rgb-self.psz, center_x_lwir_rgb+self.psz}}]
        self.val_right_lwir_rgb[i] = self.rval[img_id_lwir_rgb - num_tr_img + 1][{{}, {right_center_y_lwir_rgb-self.psz, right_center_y_lwir_rgb+self.psz}, {right_center_x_lwir_rgb-self.psz-self.half_range, right_center_x_lwir_rgb+self.psz+self.half_range}}]
    end
    collectgarbage()
end

function TrainDataHandler:next_batch()
    for idx = 1, self.batch_size do
        local i = self.tr_ptr + idx
        if i > torch.numel(self.tr_perm) then
            i = 1
            self.tr_ptr = -idx + 1
            self.curr_epoch = self.curr_epoch + 1
            print('....epoch id: ' .. self.curr_epoch .. ' done ......\n')
        end
        
        local img_id_rgb_lwir, loc_type_rgb_lwir, center_x_rgb_lwir, center_y_rgb_lwir, right_center_x_rgb_lwir, img_id_lwir_rgb, loc_type_lwir_rgb, center_x_lwir_rgb, center_y_lwir_rgb, right_center_x_lwir_rgb
        img_id_rgb_lwir, loc_type_rgb_lwir, center_x_rgb_lwir, center_y_rgb_lwir, right_center_x_rgb_lwir = self.tr_loc[self.tr_perm[i]][1], self.tr_loc[self.tr_perm[i]][2], self.tr_loc[self.tr_perm[i]][3], self.tr_loc[self.tr_perm[i]][4], self.tr_loc[self.tr_perm[i]][5]
        img_id_lwir_rgb, loc_type_lwir_rgb, right_center_x_lwir_rgb, center_y_lwir_rgb, center_x_lwir_rgb = self.tr_loc[self.tr_perm[i]][1], self.tr_loc[self.tr_perm[i]][2], self.tr_loc[self.tr_perm[i]][3], self.tr_loc[self.tr_perm[i]][4], self.tr_loc[self.tr_perm[i]][5]

        local right_center_y_rgb_lwir = center_y_rgb_lwir
        local right_center_y_lwir_rgb = center_y_lwir_rgb

        -if loc_type_rgb_lwir == 1 then
            self.batch_left_rgb_lwir[idx] = self.ldata[img_id_rgb_lwir][{{}, {center_y_rgb_lwir-self.psz, center_y_rgb_lwir+self.psz}, {center_x_rgb_lwir-self.psz, center_x_rgb_lwir+self.psz}}]
            self.batch_right_rgb_lwir[idx] = self.rdata[img_id_rgb_lwir][{{}, {right_center_y_rgb_lwir-self.psz, right_center_y_rgb_lwir+self.psz}, {right_center_x_rgb_lwir-self.psz-self.half_range, right_center_x_rgb_lwir+self.psz+self.half_range}}]
            self.batch_left_lwir_rgb[idx] = self.rdata[img_id_lwir_rgb][{{}, {center_y_lwir_rgb-self.psz, center_y_lwir_rgb+self.psz}, {center_x_lwir_rgb-self.psz, center_x_lwir_rgb+self.psz}}]
            self.batch_right_lwir_rgb[idx] = self.ldata[img_id_lwir_rgb][{{}, {right_center_y_lwir_rgb-self.psz, right_center_y_lwir_rgb+self.psz}, {right_center_x_lwir_rgb-self.psz-self.half_range, right_center_x_lwir_rgb+self.psz+self.half_range}}]
    end

    self.tr_ptr = self.tr_ptr + self.batch_size
    
    if self.cuda == 1 then
        return self.batch_left_rgb_lwir:cuda(), self.batch_right_rgb_lwir:cuda(), self.batch_label:cuda(), self.batch_left_lwir_rgb:cuda(), self.batch_right_lwir_rgb:cuda(), self.batch_label:cuda()
    else
        return self.batch_left, self.batch_right, self.batch_label
    end

end 

function TrainDataHandler:get_eval_cuda()
    return self.val_left_rgb_lwir:cuda(), self.val_right_rgb_lwir:cuda(), self.val_label:cuda(), self.val_left_lwir_rgb:cuda(), self.val_right_lwir_rgb:cuda(), self.val_label:cuda()
end

