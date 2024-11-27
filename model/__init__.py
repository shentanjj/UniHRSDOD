from .segmenter import UniLSeg

def build_segmenter(cfg):
    model = UniLSeg(cfg)
    return model
