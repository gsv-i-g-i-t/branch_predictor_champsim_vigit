#include "hashed_perceptron_v3.h"

bool hashed_perceptron_v3::predict_branch(champsim::address pc)
{
  auto pc_index = pc.slice_lower<TABLE_INDEX_BITS>().to<uint64_t>();
  result_t res;

  // ---- bias
  res.indices[0] = pc_index & (TABLE_SIZE - 1);

  // ---- global history tables
  for (std::size_t i = 0; i < GLOBAL_TABLES; i++) {
    uint64_t h = ghist_words[i].value();
    uint64_t idx = (h * (0x9e3779b97f4a7c15ULL + i))
                 ^ (pc_index * (0xC2B2AE3D27D4EB4FULL + i));
    res.indices[i+1] = idx & (TABLE_SIZE - 1);
  }

  // ---- local history table
  uint64_t local_idx = pc_index & (LOCAL_SIZE - 1);
  uint64_t local_hist = local_history[local_idx];
  res.indices[TOTAL_TABLES-1] = local_hist & (TABLE_SIZE - 1);

  // ---- compute sum
  res.yout = 0;
  for (std::size_t i = 0; i < TOTAL_TABLES; i++)
    res.yout += tables[i][res.indices[i]].value();

  last_result = res;
  return res.yout >= 0;
}

void hashed_perceptron_v3::last_branch_result(champsim::address pc,
                                              champsim::address,
                                              bool taken,
                                              uint8_t)
{
  int t = taken ? 1 : -1;

  // update global history
  for (auto& h : ghist_words)
    h.push_back(taken);

  // update local history
  uint64_t pc_index = pc.slice_lower<TABLE_INDEX_BITS>().to<uint64_t>();
  uint64_t local_idx = pc_index & (LOCAL_SIZE - 1);
  local_history[local_idx] =
      ((local_history[local_idx] << 1) | taken) & 0xFFF;

  bool correct = (taken == (last_result.yout >= 0));
  bool weak = std::abs(last_result.yout) < theta + 10;

  if (!correct || weak) {
    for (std::size_t i = 0; i < TOTAL_TABLES; i++)
      tables[i][last_result.indices[i]] += t;

    adjust_threshold(correct);
  }

  // small weight decay every few updates
  static int decay_counter = 0;
  if (++decay_counter % 100000 == 0) {
    for (auto& table : tables)
      for (auto& w : table)
        w += -(w.value() >> 5);
  }
}

void hashed_perceptron_v3::adjust_threshold(bool correct)
{
  constexpr int SPEED = 32;

  if (!correct) {
    tc++;
    if (tc >= SPEED) {
      theta++;
      tc = 0;
    }
  } else {
    tc--;
    if (tc <= -SPEED) {
      theta--;
      tc = 0;
    }
  }
}
