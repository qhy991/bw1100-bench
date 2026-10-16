"""Resolve the starter without replacing the frozen Task-owned Workload."""
from importlib import import_module


def task_source(task, contract):
    from open_cake_ir.tasks.workloads import create_task

    original = contract['workload']
    if task in ('aka_nearest_scatter', 'aka_maxpool3d_backward'):
        from open_cake_ir.tasks.workloads import WorkloadContract, validate_workload_document

        validate_workload_document(original)
        owner = import_module('open_cake_ir.tasks.' + task)
        workload = WorkloadContract(original)
        if original['operator'] != owner.TASK or workload.target != 'gfx938':
            raise ValueError('Explicit Workload differs from its Task or Target')
        return original, owner.starter_source(workload, 'primary')
    document, starter = create_task(task, backend='triton-dcu', **{
        key: contract[key] for key in ('rows', 'columns', 'depth')})
    if document != original:
        raise ValueError(task + ': original whole Workload changed')
    return document, starter
