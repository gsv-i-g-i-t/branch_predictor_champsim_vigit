#include "idpn_v3.h"

int idpn_v3::clamp(int w) {
  if (w > MAX_W) return MAX_W;
  if (w < MIN_W) return MIN_W;
  return w;
}

std::size_t idpn_v3::hash(champsim::address ip, int hist_len, int table_id) {

  std::size_t h = ip.to<std::size_t>() ^ (table_id * 1315423911);

  for (int i = 0; i < hist_len; i += 32) {
    uint32_t chunk = 0;
    for (int b = 0; b < 32 && (i + b) < GHIST; b++)
      if (ghistory[i + b])
        chunk |= (1u << b);

    h ^= chunk;
    h = (h << 5) | (h >> 59);
  }

  return h % TABLE_SIZE;
}

int idpn_v3::compute_sum(champsim::address ip)
{
  int sum = 0;

  int history_lengths[NUM_TABLES] = {8, 32, 128, 256};

  for (int t = 0; t < NUM_TABLES; t++) {
    auto idx = hash(ip, history_lengths[t], t);
    sum += tables[t].weights[idx];
  }

  sum += bias_table[ip.to<std::size_t>() % TABLE_SIZE];
  sum += corrector_table[hash(ip, 16, 99)];

  return sum;
}

bool idpn_v3::predict_branch(champsim::address ip)
{
  return compute_sum(ip) >= 0;
}

void idpn_v3::last_branch_result(champsim::address ip,
                                 champsim::address,
                                 bool taken,
                                 uint8_t)
{
  int sum = compute_sum(ip);
  bool prediction = sum >= 0;
  int target = taken ? 1 : -1;

  if (prediction != taken || std::abs(sum) < THRESH) {

    int history_lengths[NUM_TABLES] = {8, 32, 128, 256};

    for (int t = 0; t < NUM_TABLES; t++) {
      auto idx = hash(ip, history_lengths[t], t);
      tables[t].weights[idx] =
          clamp(tables[t].weights[idx] + target);
    }

    auto bidx = ip.to<std::size_t>() % TABLE_SIZE;
    bias_table[bidx] = clamp(bias_table[bidx] + target);

    auto cidx = hash(ip, 16, 99);
    corrector_table[cidx] =
        clamp(corrector_table[cidx] + target);
  }

  ghistory <<= 1;
  ghistory[0] = taken;

  for (int i = PATH_HIST - 1; i > 0; i--)
    path_history[i] = path_history[i - 1];

  path_history[0] = ip.to<uint32_t>();
}
