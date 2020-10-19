from visdialch.encoders.lf import LateFusionEncoder
from visdialch.encoders.mn import MemoryNetworkEncoder

def Encoder(model_config, *args):
    name_enc_map = {
        'lf': LateFusionEncoder,
        'mn': MemoryNetworkEncoder
    }
    return name_enc_map[model_config["encoder"]](model_config, *args)
