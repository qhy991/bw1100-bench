"""Resolve the starter without replacing the frozen Task-owned Workload."""
from importlib import import_module

# Tasks whose Workload comes from their own geometry factory instead of from
# create_task(task, rows=..., columns=..., depth=...). create_task refuses all three by
# design -- each requires launch_task.py --workload because its geometry is not a
# (rows, columns, depth) triple -- so the frozen whole Workload is validated and used
# directly. This list is the Compiler's own geometry-owner set
# (src/open_cake_ir/tasks/catalog_cli.py); keep the two in step.
GEOMETRY_TASKS = ('aka_nearest_scatter', 'aka_maxpool3d_backward',
                  'aka_broadcast_mul_gradient')


def task_source(task, contract):
    from open_cake_ir.tasks.workloads import create_task

    original = contract['workload']
    if task in GEOMETRY_TASKS:
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
