
def warmup_layers(args):
    # FIXME: distinguish dense and moe layers
    num_layers = args.num_layers
    pp = args.pipeline_model_parallel_size
    vp_chunk_size = args.num_layers_per_virtual_pipeline_stage

    vp = num_layers // pp // vp_chunk_size
    cnt = (vp - 1) * pp + (pp - 1) * 2 + 1

    return cnt * vp_chunk_size

