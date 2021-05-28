def connect(nn):

    if(nn == 0): import neuralnet.net00_pipgcn_navg as nn
    elif(nn == 1): import neuralnet.net01_pipgcn_neavg as nn
    elif(nn == 2): import neuralnet.net02_pipgcn_odepn as nn

    return nn
