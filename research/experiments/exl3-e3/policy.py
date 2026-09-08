# SPDX-License-Identifier: Apache-2.0
"""CPU metadata policy; never reads a device tensor to choose a kernel."""


def concurrent_prefill(metadata):
    # The adapter reserves E3 scratch separately during dummy profiling and
    # then runs the reference path, accounting for both persistent allocations.
    if metadata is None:
        return False
    if not isinstance(metadata, dict):
        # DBO microbatch lists need a separately qualified selection rule.
        return False
    for item in metadata.values():
        decodes = getattr(item, 'num_decodes', None)
        prefills = getattr(item, 'num_prefills', None)
        if isinstance(decodes, int) and isinstance(prefills, int):
            if prefills > 0 and decodes + prefills > 1:
                return True
    return False
