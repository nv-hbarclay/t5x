import os
import jax

from t5x.cuda_bind import cudaProfilerStart, cudaProfilerStop

collect_trace = os.environ.get('COLLECT_TRACE', '0') == '1'
use_jax_profiler = os.environ.get('USE_JAX_PROFILER', '0') == '1'
jax_profile_dir = os.environ.get('JAX_PROFILE_DIR', '/tmp/jax_trace')
start_step = int(os.environ.get('START_STEP', '10'))
stop_step = int(os.environ.get('STOP_STEP', '11'))
   
def startOrStopProfile(cur_step: int):
  if not collect_trace:
      return

  if cur_step == start_step:
    if use_jax_profiler:
      jax.profiler.start_trace(jax_profile_dir)
    else:
      cudaProfilerStart()
  
  if cur_step == stop_step:
    if use_jax_profiler:
      jax.profiler.stop_trace()
    else:
      cudaProfilerStop()
