from .segmenter import UniHRSOD

def build_segmenter(cfg):
    model = UniHRSOD(cfg)
    return model
