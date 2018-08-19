import os,sys
import torch.nn
import darknet_util
import numpy as np

class EmptyLayer(torch.nn.Module):
    def __init__(self):
        super(EmptyLayer, self).__init__()

def parseCfg(cfg_file):
    f = open(cfg_file)
    lines = f.readlines()
    f.close()

    modules = []
    module = {}

    for line in lines:
        line = line.strip('\n')
        if line == '':
            continue
        if line.startswith('#'):
            continue
        if line.startswith('['):
            if len(module) > 0:
                modules.append(module)
                module = {}
            module['type'] = line[1:-1]
            continue
        line_arr = line.split('=')
        if len(line_arr) <= 1:
            print(line)
        module[line_arr[0].strip()] = line_arr[1].strip()
    if len(module) > 0:
        modules.append(module)
    return modules

def createConvModule(module_dict, index, prev_channels):
    module = torch.nn.Sequential()
    padding = 0
    if 'batch_normalize' in module_dict:
        conv_bias = False 
        has_bn = True
    else:
        conv_bias = True 
        has_bn = False
    if int(module_dict['pad']) == 1:
        padding = (int(module_dict['size']) - 1) / 2

    filters = int(module_dict['filters'])

    ## add conv
    conv = torch.nn.Conv2d(prev_channels,
                           filters,
                           int(module_dict['size']),
                           int(module_dict['stride']),
                           padding,
                           bias=conv_bias)
    module.add_module('conv_{}'.format(index), conv)

    # add batch normalization
    if has_bn:
        bn = torch.nn.BatchNorm2d(filters)
        module.add_module('barch_norm_{}'.format(index), bn)

    ## add activation layer
    if module_dict['activation'] == 'leaky':
        activation = torch.nn.LeakyReLU(0.1, inplace=True)
        module.add_module('activation_{}'.format(index), activation)
    return filters, module


def createShortCutModule(module_dict, index, prev_channels):
    module = torch.nn.Sequential()
    empty_layer = EmptyLayer()
    module.add_module('short_cut_{}'.format(index), empty_layer)
    return prev_channels, module

def createRouteModule(module_dict, index, output_filters):
    module = torch.nn.Sequential()
    layers_arr = module_dict['layers'].split(',')
    empty_layer = EmptyLayer()
    module.add_module('route_{}'.format(index), empty_layer)
    
    output_filter = 0
    for layer in layers_arr:
        layer = int(layer)
        output_filter += output_filters[layer]
    return output_filter, module

def createUpsampleModule(module_dict, index, prev_channels):
    module = torch.nn.Sequential()
    scale_factor = int(module_dict['stride'])
    upsample = torch.nn.Upsample(scale_factor=scale_factor, mode='bilinear')
    module.add_module('upsample_{}'.format(index), upsample)
    return prev_channels, module

def createYoloModule(module_dict, index, original_img_size, use_cuda):
    mask = module_dict['mask']
    mask = mask.strip(' ').split(',')
    mask = [int(i) for i in mask]
    anchors = module_dict['anchors'].strip(' ').split(', ')
    anchors = [anchors[i] for i in mask]
    anchors = [[int(j) for j in i.strip(' ').split(',')] for i in anchors]
    #print(anchors)
    module = torch.nn.Sequential()
    yolo = darknet_util.YoloLayer(original_img_size, anchors, use_cuda)
    module.add_module('yolo_{}'.format(index), yolo)
    return module

def createModules(modules_list, use_cuda):
    net_info = modules_list[0]

    torch_module_list = torch.nn.ModuleList()

    prev_channels = 3
    output_filters = []
    for (index, module_dict) in enumerate(modules_list[1:]):
        if module_dict['type'] == 'convolutional':
            (output_filter, module) = createConvModule(module_dict, 
                                                       index,
                                                       prev_channels)
        elif module_dict['type'] == 'shortcut': 
            (output_filter, module) = createShortCutModule(module_dict,
                                                           index,
                                                           prev_channels)
        elif module_dict['type'] == 'route':
            (output_filter, module) = createRouteModule(module_dict,
                                                        index,
                                                        output_filters)
        elif module_dict['type'] == 'upsample':
            (output_filter, module) = createUpsampleModule(module_dict,
                                                           index,
                                                           prev_channels)
        elif module_dict['type'] == 'yolo':
            output_filter = -1 
            original_img_size = [int(net_info['height']), int(net_info['width'])]
            module = createYoloModule(module_dict, index, original_img_size, use_cuda)
        else:
            print('failed to create nn.module for type: {}'
                  .format(module_dict['type']))
        torch_module_list.append(module)
        prev_channels = output_filter
        output_filters.append(output_filter)
    return net_info, torch_module_list


class DarknetYoloV3(torch.nn.Module):
    def __init__(self, config_file):
        super(DarknetYoloV3, self).__init__()
        self.module_dict_list = parseCfg('./yolov3.cfg')
        use_cuda = torch.cuda.is_available()
        self.net_info, self.module_list = createModules(self.module_dict_list, use_cuda)

    def forward(self, x):
        output = []
        detections = None
        for (index, module_dict) in enumerate(self.module_dict_list[1:]):
            module_type = module_dict['type']
            if module_type == 'convolutional' or module_type == 'upsample':
                x = self.module_list[index](x)
            elif module_type == 'shortcut':
                x = output[-1] + output[int(module_dict['from'])]
            elif module_type == 'route':
                layers_arr = module_dict['layers'].split(',')
                layers_arr = [int(i) for i in layers_arr]
                x = torch.cat([output[i] for i in layers_arr], 1)
            elif module_type == 'yolo':
                if detections is None:
                    detections = self.module_list[index](x)
                else:
                    detections = torch.cat((detections, self.module_list[index](x)), 1)
                x = None
            else:
                print('failed to forward layer: {}'.format(module_type))
            output.append(x)
        return detections

    def load_weight(self, weightfile):
	#Open the weights file
        fp = open(weightfile, "rb")
    
        #The first 5 values are header information 
        # 1. Major version number
        # 2. Minor Version Number
        # 3. Subversion number 
        # 4,5. Images seen by the network (during training)
        header = np.fromfile(fp, dtype = np.int32, count = 5)
        self.header = torch.from_numpy(header)
        self.seen = self.header[3]   
        
        weights = np.fromfile(fp, dtype = np.float32)
        
        ptr = 0
        for i in range(len(self.module_list)):
            module_type = self.module_dict_list[i + 1]["type"]
    
            #If module_type is convolutional load weights
            #Otherwise ignore.
            
            if module_type == "convolutional":
                model = self.module_list[i]
                try:
                    batch_normalize = int(self.module_dict_list[i+1]["batch_normalize"])
                except:
                    batch_normalize = 0
            
                conv = model[0]
                
                
                if (batch_normalize):
                    bn = model[1]
        
                    #Get the number of weights of Batch Norm Layer
                    num_bn_biases = bn.bias.numel()
        
                    #Load the weights
                    bn_biases = torch.from_numpy(weights[ptr:ptr + num_bn_biases])
                    ptr += num_bn_biases
        
                    bn_weights = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr  += num_bn_biases
        
                    bn_running_mean = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr  += num_bn_biases
        
                    bn_running_var = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr  += num_bn_biases
        
                    #Cast the loaded weights into dims of model weights. 
                    bn_biases = bn_biases.view_as(bn.bias.data)
                    bn_weights = bn_weights.view_as(bn.weight.data)
                    bn_running_mean = bn_running_mean.view_as(bn.running_mean)
                    bn_running_var = bn_running_var.view_as(bn.running_var)
        
                    #Copy the data to model
                    bn.bias.data.copy_(bn_biases)
                    bn.weight.data.copy_(bn_weights)
                    bn.running_mean.copy_(bn_running_mean)
                    bn.running_var.copy_(bn_running_var)
                
                else:
                    #Number of biases
                    num_biases = conv.bias.numel()
                
                    #Load the weights
                    conv_biases = torch.from_numpy(weights[ptr: ptr + num_biases])
                    ptr = ptr + num_biases
                
                    #reshape the loaded weights according to the dims of the model weights
                    conv_biases = conv_biases.view_as(conv.bias.data)
                
                    #Finally copy the data
                    conv.bias.data.copy_(conv_biases)
                    
                #Let us load the weights for the Convolutional layers
                num_weights = conv.weight.numel()
                
                #Do the same as above for weights
                conv_weights = torch.from_numpy(weights[ptr:ptr+num_weights])
                ptr = ptr + num_weights
                
                conv_weights = conv_weights.view_as(conv.weight.data)
                conv.weight.data.copy_(conv_weights)

if __name__=='__main__':
    modules_list = parseCfg('./yolov3.cfg')
    use_cuda = torch.cuda.is_available()
    net_info, modules = createModules(modules_list, use_cuda)
    #print('hello world')
