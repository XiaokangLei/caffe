import caffe

import numpy as np


np.set_printoptions(threshold='nan')


MODEL_FILE = 'train_val.prototxt'

PRETRAIN_FILE = 'solver_iter_500.caffemodel'


params_txt = 'params.txt'
pf = open(params_txt, 'w')


net = caffe.Net(MODEL_FILE, PRETRAIN_FILE, caffe.TEST)

for param_name in net.params.keys():
    print(param_name)
    print(len(net.params[param_name]))
#conv1
    if param_name=='conv1':
      weight = net.params[param_name][0].data
      pf.write(param_name)
      pf.write('\n')
      pf.write('\n' + param_name + '_weight:\n\n')
    
      weight.shape = (-1, 1)

      for w in weight:
         pf.write('%ff, ' % w)
    if param_name=='conv1_scale':
     weight = net.params[param_name][0].data
     bias = net.params[param_name][1].data

    
     pf.write(param_name)
     pf.write('\n')

    
     pf.write('\n' + param_name + '_weight:\n\n')
    
     weight.shape = (-1, 1)

     for w in weight:
        pf.write('%ff, ' % w)
   
     pf.write('\n\n' + param_name + '_bias:\n\n')
    
     bias.shape = (-1, 1)
     for b in bias:
        pf.write('%ff, ' % b)
    if param_name=='conv1_bn':
     x = net.params[param_name][0].data
     y = net.params[param_name][1].data
     z = net.params[param_name][2].data
    
     pf.write(param_name)
     pf.write('\n')

    
     pf.write('\n' + param_name + '_x:\n\n')
    
     x.shape = (-1, 1)

     for w in x:
        pf.write('%ff, ' % w)
   
     pf.write('\n\n' + param_name + '_y:\n\n')
    
     y.shape = (-1, 1)
     for b in y:
        pf.write('%ff, ' % b)
     pf.write('\n\n' + param_name + '_z:\n\n')
    
     z.shape = (-1, 1)
     for c in z:
        pf.write('%ff, ' % c)
#conv2
    if param_name=='conv2':
      weight = net.params[param_name][0].data
      pf.write(param_name)
      pf.write('\n')
      pf.write('\n' + param_name + '_weight:\n\n')
    
      weight.shape = (-1, 1)

      for w in weight:
         pf.write('%ff, ' % w)
    
    if param_name=='conv2_scale':
     weight = net.params[param_name][0].data
     bias = net.params[param_name][1].data

    
     pf.write(param_name)
     pf.write('\n')

    
     pf.write('\n' + param_name + '_weight:\n\n')
    
     weight.shape = (-1, 1)

     for w in weight:
        pf.write('%ff, ' % w)
   
     pf.write('\n\n' + param_name + '_bias:\n\n')
    
     bias.shape = (-1, 1)
     for b in bias:
        pf.write('%ff, ' % b)
    if param_name=='conv2_bn':
     x = net.params[param_name][0].data
     y = net.params[param_name][1].data
     z = net.params[param_name][2].data
    
     pf.write(param_name)
     pf.write('\n')

    
     pf.write('\n' + param_name + '_x:\n\n')
    
     x.shape = (-1, 1)

     for w in x:
        pf.write('%ff, ' % w)
   
     pf.write('\n\n' + param_name + '_y:\n\n')
    
     y.shape = (-1, 1)
     for b in y:
        pf.write('%ff, ' % b)
     pf.write('\n\n' + param_name + '_z:\n\n')
    
     z.shape = (-1, 1)
     for c in z:
        pf.write('%ff, ' % c)
#conv3
     if param_name=='conv3':
      weight = net.params[param_name][0].data
      pf.write(param_name)
      pf.write('\n')
      pf.write('\n' + param_name + '_weight:\n\n')
    
      weight.shape = (-1, 1)

      for w in weight:
         pf.write('%ff, ' % w)
    if param_name=='conv3_scale':
     weight = net.params[param_name][0].data
     bias = net.params[param_name][1].data

    
     pf.write(param_name)
     pf.write('\n')

    
     pf.write('\n' + param_name + '_weight:\n\n')
    
     weight.shape = (-1, 1)

     for w in weight:
        pf.write('%ff, ' % w)
   
     pf.write('\n\n' + param_name + '_bias:\n\n')
    
     bias.shape = (-1, 1)
     for b in bias:
        pf.write('%ff, ' % b)
    if param_name=='conv3_bn':
     x = net.params[param_name][0].data
     y = net.params[param_name][1].data
     z = net.params[param_name][2].data
    
     pf.write(param_name)
     pf.write('\n')

    
     pf.write('\n' + param_name + '_x:\n\n')
    
     x.shape = (-1, 1)

     for w in x:
        pf.write('%ff, ' % w)
   
     pf.write('\n\n' + param_name + '_y:\n\n')
    
     y.shape = (-1, 1)
     for b in y:
        pf.write('%ff, ' % b)
     pf.write('\n\n' + param_name + '_z:\n\n')
    
     z.shape = (-1, 1)
     for c in z:
        pf.write('%ff, ' % c)
#conv4
    if param_name=='conv4':
      weight = net.params[param_name][0].data
      pf.write(param_name)
      pf.write('\n')
      pf.write('\n' + param_name + '_weight:\n\n')
    
      weight.shape = (-1, 1)

      for w in weight:
         pf.write('%ff, ' % w)
    if param_name=='conv4_scale':
     weight = net.params[param_name][0].data
     bias = net.params[param_name][1].data

    
     pf.write(param_name)
     pf.write('\n')

    
     pf.write('\n' + param_name + '_weight:\n\n')
    
     weight.shape = (-1, 1)

     for w in weight:
        pf.write('%ff, ' % w)
   
     pf.write('\n\n' + param_name + '_bias:\n\n')
    
     bias.shape = (-1, 1)
     for b in bias:
        pf.write('%ff, ' % b)
    if param_name=='conv4_bn':
     x = net.params[param_name][0].data
     y = net.params[param_name][1].data
     z = net.params[param_name][2].data
    
     pf.write(param_name)
     pf.write('\n')

    
     pf.write('\n' + param_name + '_x:\n\n')
    
     x.shape = (-1, 1)

     for w in x:
        pf.write('%ff, ' % w)
   
     pf.write('\n\n' + param_name + '_y:\n\n')
    
     y.shape = (-1, 1)
     for b in y:
        pf.write('%ff, ' % b)
     pf.write('\n\n' + param_name + '_z:\n\n')
    
     z.shape = (-1, 1)
     for c in z:
        pf.write('%ff, ' % c)
#conv5
    if param_name=='conv5':
      weight = net.params[param_name][0].data
      pf.write(param_name)
      pf.write('\n')
      pf.write('\n' + param_name + '_weight:\n\n')
    
      weight.shape = (-1, 1)

      for w in weight:
         pf.write('%ff, ' % w)
    if param_name=='conv5_scale':
     weight = net.params[param_name][0].data
     bias = net.params[param_name][1].data

    
     pf.write(param_name)
     pf.write('\n')

    
     pf.write('\n' + param_name + '_weight:\n\n')
    
     weight.shape = (-1, 1)

     for w in weight:
        pf.write('%ff, ' % w)
   
     pf.write('\n\n' + param_name + '_bias:\n\n')
    
     bias.shape = (-1, 1)
     for b in bias:
        pf.write('%ff, ' % b)
    if param_name=='conv5_bn':
     x = net.params[param_name][0].data
     y = net.params[param_name][1].data
     z = net.params[param_name][2].data
    
     pf.write(param_name)
     pf.write('\n')

    
     pf.write('\n' + param_name + '_x:\n\n')
    
     x.shape = (-1, 1)

     for w in x:
        pf.write('%ff, ' % w)
   
     pf.write('\n\n' + param_name + '_y:\n\n')
    
     y.shape = (-1, 1)
     for b in y:
        pf.write('%ff, ' % b)
     pf.write('\n\n' + param_name + '_z:\n\n')
    
     z.shape = (-1, 1)
     for c in z:
        pf.write('%ff, ' % c)
#conv6
    if param_name=='conv6':
      weight = net.params[param_name][0].data
      pf.write(param_name)
      pf.write('\n')
      pf.write('\n' + param_name + '_weight:\n\n')
    
      weight.shape = (-1, 1)

      for w in weight:
         pf.write('%ff, ' % w)
    if param_name=='conv6_scale':
     weight = net.params[param_name][0].data
     bias = net.params[param_name][1].data

    
     pf.write(param_name)
     pf.write('\n')

    
     pf.write('\n' + param_name + '_weight:\n\n')
    
     weight.shape = (-1, 1)

     for w in weight:
        pf.write('%ff, ' % w)
   
     pf.write('\n\n' + param_name + '_bias:\n\n')
    
     bias.shape = (-1, 1)
     for b in bias:
        pf.write('%ff, ' % b)
    if param_name=='conv6_bn':
     x = net.params[param_name][0].data
     y = net.params[param_name][1].data
     z = net.params[param_name][2].data
    
     pf.write(param_name)
     pf.write('\n')

    
     pf.write('\n' + param_name + '_x:\n\n')
    
     x.shape = (-1, 1)

     for w in x:
        pf.write('%ff, ' % w)
   
     pf.write('\n\n' + param_name + '_y:\n\n')
    
     y.shape = (-1, 1)
     for b in y:
        pf.write('%ff, ' % b)
     pf.write('\n\n' + param_name + '_z:\n\n')
    
     z.shape = (-1, 1)
     for c in z:
        pf.write('%ff, ' % c)
#conv7
    if param_name=='conv7':
      weight = net.params[param_name][0].data
      pf.write(param_name)
      pf.write('\n')
      pf.write('\n' + param_name + '_weight:\n\n')
    
      weight.shape = (-1, 1)

      for w in weight:
         pf.write('%ff, ' % w)
    if param_name=='conv7_scale':
     weight = net.params[param_name][0].data
     bias = net.params[param_name][1].data

    
     pf.write(param_name)
     pf.write('\n')

    
     pf.write('\n' + param_name + '_weight:\n\n')
    
     weight.shape = (-1, 1)

     for w in weight:
        pf.write('%ff, ' % w)
   
     pf.write('\n\n' + param_name + '_bias:\n\n')
    
     bias.shape = (-1, 1)
     for b in bias:
        pf.write('%ff, ' % b)
    if param_name=='conv7_bn':
     x = net.params[param_name][0].data
     y = net.params[param_name][1].data
     z = net.params[param_name][2].data
    
     pf.write(param_name)
     pf.write('\n')

    
     pf.write('\n' + param_name + '_x:\n\n')
    
     x.shape = (-1, 1)

     for w in x:
        pf.write('%ff, ' % w)
   
     pf.write('\n\n' + param_name + '_y:\n\n')
    
     y.shape = (-1, 1)
     for b in y:
        pf.write('%ff, ' % b)
     pf.write('\n\n' + param_name + '_z:\n\n')
    
     z.shape = (-1, 1)
     for c in z:
        pf.write('%ff, ' % c)
#conv8
    if param_name=='conv8':
      weight = net.params[param_name][0].data
      pf.write(param_name)
      pf.write('\n')
      pf.write('\n' + param_name + '_weight:\n\n')
    
      weight.shape = (-1, 1)

      for w in weight:
         pf.write('%ff, ' % w)
    if param_name=='conv8_scale':
     weight = net.params[param_name][0].data
     bias = net.params[param_name][1].data

    
     pf.write(param_name)
     pf.write('\n')

    
     pf.write('\n' + param_name + '_weight:\n\n')
    
     weight.shape = (-1, 1)

     for w in weight:
        pf.write('%ff, ' % w)
   
     pf.write('\n\n' + param_name + '_bias:\n\n')
    
     bias.shape = (-1, 1)
     for b in bias:
        pf.write('%ff, ' % b)
    if param_name=='conv8_bn':
     x = net.params[param_name][0].data
     y = net.params[param_name][1].data
     z = net.params[param_name][2].data
    
     pf.write(param_name)
     pf.write('\n')

    
     pf.write('\n' + param_name + '_x:\n\n')
    
     x.shape = (-1, 1)

     for w in x:
        pf.write('%ff, ' % w)
   
     pf.write('\n\n' + param_name + '_y:\n\n')
    
     y.shape = (-1, 1)
     for b in y:
        pf.write('%ff, ' % b)
     pf.write('\n\n' + param_name + '_z:\n\n')
    
     z.shape = (-1, 1)
     for c in z:
        pf.write('%ff, ' % c)
#fc
    if param_name=='fc9':
     weight = net.params[param_name][0].data
     bias = net.params[param_name][1].data

    
     pf.write(param_name)
     pf.write('\n')

    
     pf.write('\n' + param_name + '_weight:\n\n')
    
     weight.shape = (-1, 1)

     for w in weight:
        pf.write('%ff, ' % w)
   
     pf.write('\n\n' + param_name + '_bias:\n\n')
    
     bias.shape = (-1, 1)
     for b in bias:
        pf.write('%ff, ' % b)
    

    pf.write('\n\n')

pf.close
