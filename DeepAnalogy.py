from VGG19 import VGG19
import PatchMatch as pm
import torch
import numpy as np
import copy
from utils import *
import time
import datetime
from config.settings import USE_CUDA


def prepare_image2(img_org):
    lab = cv2.cvtColor(img_org, cv2.COLOR_BGR2LAB)
    lab_scaled = lab.astype("float32")
    L,a,b = cv2.split(lab_scaled)
    L *= (100/255)
    L -= 50

    # cv2.dnn.blobFromImage is supposed to be equivalent to normalizing by
    # mean = [0.485, 0.456, 0.406],
    # std = [0.229, 0.224, 0.225].......... but double check that, maybe it is not needed at all in this situation
    # L = cv2.dnn.blobFromImage(L)

    img_A_L = np.stack((L,)*3, axis=-1)

    img_A_ab = np.zeros(shape=[img_org.shape[0],img_org.shape[1], 2])
    return img_A_L, img_A_ab


def prepare_image(img_org):
    lab = cv2.cvtColor(img_org, cv2.COLOR_BGR2LAB)
    L,a,b = cv2.split(lab)

    # cv2.dnn.blobFromImage is supposed to be equivalent to normalizing by
    # mean = [0.485, 0.456, 0.406],
    # std = [0.229, 0.224, 0.225].......... but double check that, maybe it is not needed at all in this situation
    # L = cv2.dnn.blobFromImage(L)

    L3 = np.stack((L,)*3, axis=-1)
    return L3, lab


def analogy(img_A_L, img_BP_L, config):
    img_A_L, img_A_Lab = prepare_image(img_A_L)
    img_BP_L, img_BP_Lab = prepare_image(img_BP_L)

    start_time_0 = time.time()

    weights = config['weights']
    sizes = config['sizes']
    rangee = config['rangee']
    deconv_iters = config['deconv_iters']
    params = config['params']
    lr = config['lr']

    device = torch.device("cuda" if USE_CUDA else "cpu")


    # preparing data
    img_A_tensor = torch.FloatTensor(img_A_L.transpose(2, 0, 1))
    img_BP_tensor = torch.FloatTensor(img_BP_L.transpose(2, 0, 1))
    img_A_tensor, img_BP_tensor = img_A_tensor.to(device), img_BP_tensor.to(device)

    img_A_tensor = img_A_tensor.unsqueeze(0)
    img_BP_tensor = img_BP_tensor.unsqueeze(0)

    # compute 5 feature maps
    model = VGG19(device=device)
    data_A, data_A_size = model.get_features(img_tensor=img_A_tensor.clone(), layers=params['layers'])
    data_AP = copy.deepcopy(data_A)
    data_BP, data_B_size = model.get_features(img_tensor=img_BP_tensor.clone(), layers=params['layers'])
    data_B = copy.deepcopy(data_BP)
    print("Features extracted!")

    # usually 5 layers
    n_layers = len(params['layers'])


    for curr_layer in range(n_layers):
        print("\n### current stage: %d - start ###"%(5-curr_layer))
        start_time_1 = time.time()

        if curr_layer == 0:
            ann_AB = pm.init_nnf(data_A_size[curr_layer][2:], data_B_size[curr_layer][2:])
            ann_BA = pm.init_nnf(data_B_size[curr_layer][2:], data_A_size[curr_layer][2:])
        else:
            ann_AB = pm.upSample_nnf(ann_AB, data_A_size[curr_layer][2:])
            ann_BA = pm.upSample_nnf(ann_BA, data_B_size[curr_layer][2:])

        # blend feature
        Ndata_A, response_A = normalize(data_A[curr_layer])
        Ndata_BP, response_BP = normalize(data_BP[curr_layer])

        data_AP[curr_layer] = blend(response_A, data_A[curr_layer], data_AP[curr_layer], weights[curr_layer])
        data_B[curr_layer] = blend(response_BP, data_BP[curr_layer], data_B[curr_layer], weights[curr_layer])

        Ndata_AP, _ = normalize(data_AP[curr_layer])
        Ndata_B, _ = normalize(data_B[curr_layer])

        # NNF search
        print("- NNF search for ann_AB")
        start_time_2 = time.time()
        ann_AB, _ = pm.propagate(ann_AB, ts2np(Ndata_A), ts2np(Ndata_AP), ts2np(Ndata_B), ts2np(Ndata_BP), sizes[curr_layer],
                              params['propagate_iter'], rangee[curr_layer])
        print("\tElapse: "+str(datetime.timedelta(seconds=time.time()- start_time_2))[:-7])

        print("- NNF search for ann_BA")
        start_time_2 = time.time()
        ann_BA, _ = pm.propagate(ann_BA, ts2np(Ndata_BP), ts2np(Ndata_B), ts2np(Ndata_AP), ts2np(Ndata_A), sizes[curr_layer],
                              params['propagate_iter'], rangee[curr_layer])
        print("\tElapse: "+str(datetime.timedelta(seconds=time.time()- start_time_2))[:-7])        

        if curr_layer >= 4:
            print("### current stage: %d - end | "%(5-curr_layer)+"Elapse: "+str(datetime.timedelta(seconds=time.time()- start_time_1))[:-7]+' ###')
            break

        # using backpropagation to approximate feature
        next_layer = curr_layer + 2

        ann_AB_upnnf2 = pm.upSample_nnf(ann_AB, data_A_size[next_layer][2:])
        ann_BA_upnnf2 = pm.upSample_nnf(ann_BA, data_B_size[next_layer][2:])

        data_AP_np = pm.avg_vote(ann_AB_upnnf2, ts2np(data_BP[next_layer]), sizes[next_layer], data_A_size[next_layer][2:],
                              data_B_size[next_layer][2:])
        data_B_np = pm.avg_vote(ann_BA_upnnf2, ts2np(data_A[next_layer]), sizes[next_layer], data_B_size[next_layer][2:],
                             data_A_size[next_layer][2:])

        data_AP[next_layer] = np2ts(data_AP_np, device)
        data_B[next_layer] = np2ts(data_B_np, device)

        target_BP_np = pm.avg_vote(ann_AB, ts2np(data_BP[curr_layer]), sizes[curr_layer], data_A_size[curr_layer][2:],
                                data_B_size[curr_layer][2:])
        target_A_np = pm.avg_vote(ann_BA, ts2np(data_A[curr_layer]), sizes[curr_layer], data_B_size[curr_layer][2:],
                               data_A_size[curr_layer][2:])

        target_BP = np2ts(target_BP_np, device)
        target_A = np2ts(target_A_np, device)

        print('- deconvolution for feat A\'')
        start_time_2 = time.time()
        data_AP[curr_layer+1] = model.get_deconvoluted_feat(target_BP, curr_layer, data_AP[next_layer], lr=lr[curr_layer],
                                                              iters=deconv_iters, display=False)
        print("\tElapse: "+str(datetime.timedelta(seconds=time.time()- start_time_2))[:-7])        

        print('- deconvolution for feat B')
        start_time_2 = time.time()        
        data_B[curr_layer+1] = model.get_deconvoluted_feat(target_A, curr_layer, data_B[next_layer], lr=lr[curr_layer],
                                                             iters=deconv_iters, display=False)
        print("\tElapse: "+str(datetime.timedelta(seconds=time.time()- start_time_2))[:-7])                

        if USE_CUDA:
            # in case of data type inconsistency
            if data_B[curr_layer + 1].type() == torch.cuda.DoubleTensor:
                data_B[curr_layer + 1] = data_B[curr_layer + 1].type(torch.cuda.FloatTensor)
                data_AP[curr_layer + 1] = data_AP[curr_layer + 1].type(torch.cuda.FloatTensor)

        else:
            if data_B[curr_layer + 1].type() == torch.DoubleTensor:
                data_B[curr_layer + 1] = data_B[curr_layer + 1].type(torch.FloatTensor)
                data_AP[curr_layer + 1] = data_AP[curr_layer + 1].type(torch.FloatTensor)
        
        print("### current stage: %d - end | "%(5-curr_layer)+"Elapse: "+str(datetime.timedelta(seconds=time.time()- start_time_1))[:-7]+' ###')


    print('\n- reconstruct images A\' and B')
    img_AP_Lab = pm.reconstruct_avg(ann_AB, img_BP_Lab, sizes[curr_layer], data_A_size[curr_layer][2:], data_B_size[curr_layer][2:])
    img_B_Lab = pm.reconstruct_avg(ann_BA, img_A_Lab, sizes[curr_layer], data_A_size[curr_layer][2:], data_B_size[curr_layer][2:])

    img_AP_Lab = np.clip(img_AP_Lab, 0, 255).astype("uint8")
    img_B_Lab = np.clip(img_B_Lab, 0, 255).astype("uint8")

    img_AP_L = cv2.split(img_AP_Lab)[0]
    img_B_L = cv2.split(img_B_Lab)[0]

    img_AP_bgr = cv2.cvtColor(img_AP_Lab, cv2.COLOR_LAB2BGR)
    img_B_bgr = cv2.cvtColor(img_B_Lab, cv2.COLOR_LAB2BGR)



    return img_AP_L, img_AP_bgr, img_B_L, img_B_bgr, str(datetime.timedelta(seconds=time.time()- start_time_0))[:-7]







