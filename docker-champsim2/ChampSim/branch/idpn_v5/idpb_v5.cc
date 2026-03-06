#include "idpn_v5.h"

inline int idpn_v5::clamp(int x) {
  if (x > MAX_W) return MAX_W;
  if (x < MIN_W) return MIN_W;
  return x;
}

std::size_t idpn_v5::index(champsim::address ip, int table_id) {
  return (ip.to<uint64_t>() ^ (table_id * 0x9e3779b97f4a7c15ULL))
         & (TABLE_SIZE - 1);
}

int idpn_v5::compute_segment_sum(champsim::address ip, int table_id)
{
  auto idx = index(ip, table_id);
  auto& table = tables[table_id];

  int start = 0;
  int length = 0;

  if (table_id == 0) { start = 0; length = L0; }
  if (table_id == 1) { start = L0; length = L1; }
  if (table_id == 2) { start = L0+L1; length = L2; }
  if (table_id == 3) { start = L0+L1+L2; length = L3; }

  int sum = table.bias[idx];

  for (int i = 0; i < length; i++) {
    int bit = ghistory[start + i] ? 1 : -1;
    sum += table.weights[idx][start + i] * bit;
  }

  return sum;
}

int idpn_v5::compute_total_sum(champsim::address ip)
{
  int total = 0;

  for (int t = 0; t < 4; t++)
    total += fusion_weight[t] * compute_segment_sum(ip, t);

  return total;
}

bool idpn_v5::predict_branch(champsim::address ip)
{
  return compute_total_sum(ip) >= 0;
}

void idpn_v5::last_branch_result(champsim::address ip,
                                 champsim::address,
                                 bool taken,
                                 uint8_t)
{
  int total_sum = compute_total_sum(ip);
  bool prediction = total_sum >= 0;
  int target = taken ? 1 : -1;

  int threshold = 1.5 * GHIST + 20;

  if (prediction != taken || std::abs(total_sum) < threshold) {

    for (int t = 0; t < 4; t++) {

      auto idx = index(ip, t);
      auto& table = tables[t];

      int start = 0;
      int length = 0;

      if (t == 0) { start = 0; length = L0; }
      if (t == 1) { start = L0; length = L1; }
      if (t == 2) { start = L0+L1; length = L2; }
      if (t == 3) { start = L0+L1+L2; length = L3; }

      table.bias[idx] =
        clamp(table.bias[idx] + target);

      for (int i = 0; i < length; i++) {
        int bit = ghistory[start + i] ? 1 : -1;
        table.weights[idx][start + i] =
          clamp(table.weights[idx][start + i] + target * bit);
      }
    }
  }

  // Adaptive fusion update
  for (int t = 0; t < 4; t++) {
    int seg_sum = compute_segment_sum(ip, t);
    bool seg_pred = seg_sum >= 0;

    if (seg_pred == taken)
      fusion_weight[t] = std::min(fusion_weight[t] + 1, 7);
    else
      fusion_weight[t] = std::max(fusion_weight[t] - 1, 1);
  }

  ghistory <<= 1;
  ghistory[0] = taken;
}

