import flax.nnx as nnx


kernel_init = nnx.initializers.normal(stddev=0.02)
kernel_init_out_proj = nnx.initializers.normal(stddev=0.002)
