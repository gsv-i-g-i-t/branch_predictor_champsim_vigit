#include "idpn_v4.h"

inline int idpn_v4::clamp(int x) {
  if (x > MAX_W) return MAX_W;
  if (x < MIN_W) return MIN_W;
  return x;
}

inline std::size_t idpn_v4::idx(champsim::address ip, int table_id)
{
  uint64_t ip_val = ip.to<uint64_t>();
  uint64_t mixed = ip_val ^ folded_hist ^ (table_id * 0x9e3779b97f4a7c15ULL);
  return mixed & (TABLE_SIZE - 1); // faster than %
}

int idpn_v4::compute_sum(champsim::address ip)
{
  int sum = 0;

  auto base = ip.to<uint64_t>() & (TABLE_SIZE - 1);
  sum += bias[base];

  for (int t = 0; t < NUM_TABLES; t++)
    sum += tables[t].w[idx(ip, t)];

  sum += corrector[idx(ip, 99)];

  return sum;
}

bool idpn_v4::predict_branch(champsim::address ip)
{
  return compute_sum(ip) >= 0;
}

void idpn_v4::last_branch_result(champsim::address ip,
                                 champsim::address,
                                 bool taken,
                                 uint8_t)
{
  int sum = compute_sum(ip);
  bool pred = sum >= 0;
  int target = taken ? 1 : -1;

  if (pred != taken || std::abs(sum) < THRESH) {

    auto base = ip.to<uint64_t>() & (TABLE_SIZE - 1);
    bias[base] = clamp(bias[base] + target);

    for (int t = 0; t < NUM_TABLES; t++) {
      auto i = idx(ip, t);
      tables[t].w[i] = clamp(tables[t].w[i] + target);
    }

    auto c = idx(ip, 99);
    corrector[c] = clamp(corrector[c] + target);
  }

  // Update history
  ghistory <<= 1;
  ghistory[0] = taken;

  // Fast folded history update
  folded_hist ^= (folded_hist << 7);
  folded_hist ^= (folded_hist >> 9);
  folded_hist ^= taken ? 0x9e3779b97f4a7c15ULL : 0;
}
