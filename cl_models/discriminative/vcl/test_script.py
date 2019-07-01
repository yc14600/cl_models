import os
import six
#import torch.multiprocessing as mp

#num_job = 2


seeds = [0,1,2,3,42]

split = True
permuted = False

dataset = ['mnist'] #,'fashion'
vi_type = ['KLqp_gaussian_NG']  # 'KLqp_analytic'
coreset_type = ['random','kcenter','stein']
cfg = {}

if split:
    # test split multihead tasks
    cfg['task_name'] = 'split'
    cfg['epoch'] = str(100)
    cfg['coreset_size'] = str(40)
    cfg['coreset_usage'] = 'final'
    cfg['batch_size'] = str(50000)
    cfg['local_iter'] = str(50)
    cfg['head_type'] = 'multi'
    cfg['ginit'] = str(4)

    for dt in dataset:
        cfg['dataset'] = dt
        for vt in vi_type:
            cfg['vi_type'] = vt
            for ct in coreset_type:
                cfg['coreset_type'] = ct
                if ct == 'stein':
                    cfg['local_iter'] = str(1)
                for sd in seeds:
                    cfg['seed'] = str(sd)
                    cmd_str = ':'
                    cmd_str = ' '.join([cmd_str.join(cf) for cf in six.iteritems(cfg)]) 
                    cmd_str = 'python VCL_stein.py '+cmd_str  
                    print(cmd_str)  
                    os.system(cmd_str)
if permuted:               
    # test permuted single head tasks
    cfg['task_name'] = 'permuted'
    cfg['epoch'] = str(50)
    cfg['coreset_size'] = str(200)
    cfg['coreset_usage'] = 'final'
    cfg['batch_size'] = str(500)
    cfg['local_iter'] = str(50)
    cfg['head_type'] = 'single'
    cfg['ginit'] = str(3)
    for dt in dataset:
        cfg['dataset'] = dt
        for vt in vi_type:
            cfg['vi_type'] = vt
            for ct in coreset_type:
                cfg['coreset_type'] = ct
                #if ct == 'stein':
                #    cfg['local_iter'] = str(1)
                for sd in seeds:
                    #if sd == 0 and ct == 'random' and vt == 'KLqp_analytic' and dt=='mnist':
                    #    continue
                    cfg['seed'] = str(sd)
                    cmd_str = ':'
                    cmd_str = ' '.join([cmd_str.join(cf) for cf in six.iteritems(cfg)]) 
                    cmd_str = 'python VCL_stein.py '+cmd_str  
                    print(cmd_str)  
                    #os.system(cmd_str)