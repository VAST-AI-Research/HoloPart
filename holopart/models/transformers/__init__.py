from typing import Callable, Optional

from .triposg_transformer import TripoSGDiTModel


def default_set_attn_proc_func(
    name: str,
    hidden_size: int,
    cross_attention_dim: Optional[int],
    ori_attn_proc: object,
) -> object:
    return ori_attn_proc


def set_transformer_attn_processor(
    transformer: TripoSGDiTModel,
    set_self_attn_proc_func: Callable = default_set_attn_proc_func,
    set_cross_attn_proc_func: Callable = default_set_attn_proc_func,
) -> None:
    attn_procs = {}
    for name, attn_processor in transformer.attn_processors.items():
        hidden_size = transformer.config.width
        if name.endswith("attn1.processor"):
            # self attention
            attn_procs[name] = set_self_attn_proc_func(
                name, hidden_size, None, attn_processor
            )
        elif name.endswith("attn2.processor"):
            # cross attention
            cross_attention_dim = transformer.config.cross_attention_dim
            attn_procs[name] = set_cross_attn_proc_func(
                name, hidden_size, cross_attention_dim, attn_processor
            )
        elif name.endswith("attn2_2.processor"):
            # cross attention 2
            cross_attention_dim = transformer.config.cross_attention_2_dim
            attn_procs[name] = set_cross_attn_proc_func(
                name, hidden_size, cross_attention_dim, attn_processor
            )

    transformer.set_attn_processor(attn_procs)
