#include "hashed_perceptron_pro.h"

bool hashed_perceptron_pro::predict_branch(champsim::address pc)
{
  auto pc_index = pc.slice_lower<TABLE_INDEX_BITS>().to<uint64_t>();

  result_t res;
  res.yout = 0;

  // ----- Standard geometric tables (baseline style)
  for (std::size_t i = 0; i < NTABLES; i++) {
    uint64_t h = ghist_words[i].value();
    uint64_t idx = (h ^ (pc_index * (0x9e3779b97f4a7c15ULL + i)));
    res.indices[i] = idx & (TABLE_SIZE - 1);
    res.yout += tables[i][res.indices[i]].value();
  }

  // 🔥 Reinforcement if low confidence
  if (std::abs(res.yout) < theta + 4) {
    uint64_t lh = long_history.value();
    uint64_t idx = (lh ^ (pc_index * 0xC2B2AE3D27D4EB4FULL));
    res.long_index = idx & (TABLE_SIZE - 1);
    res.yout += long_table[res.long_index].value();
  } else {
    res.long_index = 0;
  }

  last_result = res;
  return res.yout >= 0;
}

void hashed_perceptron_pro::last_branch_result(champsim::address,
                                               champsim::address,
                                               bool taken,
                                               uint8_t)
{
  int t = taken ? 1 : -1;

  for (auto& h : ghist_words)
    h.push_back(taken);

  long_history.push_back(taken);

  bool correct = (taken == (last_result.yout >= 0));
  bool weak = std::abs(last_result.yout) < theta;

  if (!correct || weak) {

    for (std::size_t i = 0; i < NTABLES; i++)
      tables[i][last_result.indices[i]] += t;

    // 🔥 Train long table only if it was used
    if (std::abs(last_result.yout) < theta + 4)
      long_table[last_result.long_index] += t;

    adjust_threshold(correct);
  }
}

void hashed_perceptron_pro::adjust_threshold(bool correct)
{
  constexpr int SPEED = 18;

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
