#include "hashed_perceptron_pl.h"

bool hashed_perceptron_pl::predict_branch(champsim::address pc)
{
  auto pc_index = pc.slice_lower<INDEX_BITS>().to<uint64_t>();

  result_t res;
  res.yout = 0;

  // Partition selection based on PC
  res.pc_partition = (pc_index >> 2) & (PC_PARTITIONS - 1);

  for (std::size_t i = 0; i < NTABLES; i++) {

    uint64_t hist = ghist_words[i].value();
    uint64_t idx = (hist ^ (pc_index * (0x9e3779b97f4a7c15ULL + i)))
                   & (TABLE_SIZE - 1);

    res.indices[i] = idx;

    res.yout += tables[i][res.pc_partition][idx].value();
  }

  last_result = res;
  return res.yout >= 0;
}

void hashed_perceptron_pl::last_branch_result(champsim::address pc,
                                              champsim::address,
                                              bool taken,
                                              uint8_t)
{
  int t = taken ? 1 : -1;

  for (auto& h : ghist_words)
    h.push_back(taken);

  bool correct = (taken == (last_result.yout >= 0));
  bool weak = std::abs(last_result.yout) < theta;

  if (!correct || weak) {
    for (std::size_t i = 0; i < NTABLES; i++)
      tables[i][last_result.pc_partition][last_result.indices[i]] += t;

    adjust_threshold(correct);
  }
}

void hashed_perceptron_pl::adjust_threshold(bool correct)
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
