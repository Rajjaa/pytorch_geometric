from yacs.config import CfgNode as CN

from torch_geometric.graphgym.register import register_config


@register_config('custom')
def set_cfg_pooling(cfg):
    """
    This function sets the default values for customized pooling layers
    """
    # ----------------------------------------------------------------------- #
    # train options
    # ----------------------------------------------------------------------- #


    # ----------------------------------------------------------------------- #
    # loader options
    # ----------------------------------------------------------------------- #

