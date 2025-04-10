# Copyright (c) OpenMMLab. All rights reserved.
import functools
from typing import Any, Callable, Iterable, Optional, Sequence, Union

import torch
from packaging import version

device_types_t = Optional[Union[str, Sequence[str]]]

WARPS_PER_SM = {
    (8, 0): 64,
    (8, 6): 48,
    (8, 7): 48,
    (8, 9): 48,
    (9, 0): 64,
    (10, 0): 64,
    (10, 1): 48,
    (12, 0): 48,
}


@functools.lru_cache
def get_device_props(device=None):
    if device is None:
        device = torch.cuda.current_device()

    props = torch.cuda.get_device_properties(device)

    warps_per_sm = WARPS_PER_SM.get((props.major, props.minor), 32)
    out = dict(
        multi_processor_count=props.multi_processor_count,
        warps_per_sm=warps_per_sm,
    )
    return out


def dummy_custom_op(
    name: str,
    fn: Optional[Callable] = None,
    /,
    *,
    mutates_args: Union[str, Iterable[str]],
    device_types: device_types_t = None,
    schema: Optional[str] = None,
) -> Any:
    return fn


def get_condition_custom_op_decorator():
    if version.parse(torch.__version__) >= version.parse('2.5.0'):
        return torch.library.custom_op
    return dummy_custom_op
