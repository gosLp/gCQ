#pragma once
#include <hip/hip_runtime.h>
#include <stdint.h>

#ifdef WF_LOG

struct WFLogEntry {
  uint64_t tid;
  uint32_t op;    // 0=enq, 1=deq
  uint64_t arg;   // enqueue value, or 0 for dequeue
  uint64_t ret;   // enq: 1=success,0=fail ; deq: value, or 0 for empty
  uint64_t t0;    // logical call timestamp
  uint64_t t1;    // logical return timestamp
};

__device__ WFLogEntry*        g_wf_log    = nullptr;
__device__ uint32_t           g_wf_stride = 0;
__device__ unsigned long long g_wf_time   = 1ull;

static __device__ __forceinline__
WFLogEntry* wf_log_slot(uint32_t tid, uint32_t& local_idx) {
  return &g_wf_log[(uint64_t)tid * (uint64_t)g_wf_stride + (uint64_t)local_idx++];
}

#define WF_LOG_ENQ_BEGIN(tid_, val_, idx_)                                      \
  WFLogEntry* __le = wf_log_slot((uint32_t)(tid_), (idx_));                     \
  __le->tid = (uint64_t)(tid_);                                                 \
  __le->op  = 0u;                                                               \
  __le->arg = (uint64_t)(val_);                                                 \
  __le->t0  = atomicAdd(&g_wf_time, 1ull);

#define WF_LOG_ENQ_END(ok_)                                                     \
  __le->ret = ((ok_) ? 1ull : 0ull);                                            \
  __le->t1  = atomicAdd(&g_wf_time, 1ull);

#define WF_LOG_DEQ_BEGIN(tid_, idx_)                                            \
  WFLogEntry* __ld = wf_log_slot((uint32_t)(tid_), (idx_));                     \
  __ld->tid = (uint64_t)(tid_);                                                 \
  __ld->op  = 1u;                                                               \
  __ld->arg = 0ull;                                                             \
  __ld->t0  = atomicAdd(&g_wf_time, 1ull);

#define WF_LOG_DEQ_END(ret_)                                                    \
  __ld->ret = (uint64_t)(ret_);                                                 \
  __ld->t1  = atomicAdd(&g_wf_time, 1ull);

#endif
