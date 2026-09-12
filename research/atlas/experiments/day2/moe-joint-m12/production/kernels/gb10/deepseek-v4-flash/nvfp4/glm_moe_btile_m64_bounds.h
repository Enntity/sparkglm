// SPDX-License-Identifier: AGPL-3.0-only
// Shared host/device concentrated-prefill bounds.
#pragma once
#ifdef __CUDACC__
#define GLM_BT64_HD __host__ __device__
#else
#define GLM_BT64_HD
#endif
constexpr unsigned glm_btile_m64_max_rows = 1088;
GLM_BT64_HD constexpr bool glm_btile_m64_work_valid(int rows, unsigned mt, unsigned nt) {
    return rows > 0 && unsigned(rows) <= glm_btile_m64_max_rows && mt < unsigned((rows + 63) / 64) && nt < 16;
}
#undef GLM_BT64_HD
