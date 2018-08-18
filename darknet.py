import os,sys
import torch.nn

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

def createYoloModule()


def createModules(modules_list):
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
            output_filter = prev_channels
            module = EmptyLayer()
        else:
            print('failed to create nn.module for type: {}'
                  .format(module_dict['type']))
        torch_module_list.add_module(module_dict['type'] + '_{}'.format(index),
                                     module)
        prev_channels = output_filter
        output_filters.append(output_filter)
    return torch_module_list

if __name__=='__main__':
    modules_list = parseCfg('./yolov3.cfg')
    modules = createModules(modules_list) 
    #print('hello world')
