#include "hashed_perceptron_elite.h"

bool hashed_perceptron_elite::predict_branch(champsim::address pc)
{
  auto pc_index = pc.slice_lower<INDEX_BITS>().to<uint64_t>();
  uint16_t pc_tag = (pc_index >> 4) & ((1 << TAG_BITS) - 1);

  result_t res;
  res.yout = 0;

  for (std::size_t i = 0; i < NTABLES; i++) {

    uint64_t hist = ghist_words[i].value();
    uint64_t idx = (hist ^ (pc_index * (0x9e3779b97f4a7c15ULL + i)))
                   & (TABLE_SIZE - 1);

    res.indices[i] = idx;

    auto& e = tables[i][idx];

    bool tag_ok = (e.tag == pc_tag);
    res.tag_match[i] = tag_ok;

    if (tag_ok) {
      int scaled = e.weight.value();

      // usefulness scaling
      scaled = scaled * (1 + e.usefulness) / 4;

      res.yout += scaled;
    }
  }

  // saturation compression
  if (res.yout > 100) res.yout = 100 + (res.yout - 100) / 2;
  if (res.yout < -100) res.yout = -100 + (res.yout + 100) / 2;

  last_result = res;
  return res.yout >= 0;
}

void hashed_perceptron_elite::last_branch_result(champsim::address pc,
                                                 champsim::address,
                                                 bool taken,
                                                 uint8_t)
{
  int t = taken ? 1 : -1;

  for (auto& h : ghist_words)
    h.push_back(taken);

  auto pc_index = pc.slice_lower<INDEX_BITS>().to<uint64_t>();
  uint16_t pc_tag = (pc_index >> 4) & ((1 << TAG_BITS) - 1);

  bool correct = (taken == (last_result.yout >= 0));
  bool weak = std::abs(last_result.yout) < theta;

  if (!correct || weak) {

    for (std::size_t i = 0; i < NTABLES; i++) {

      auto& e = tables[i][last_result.indices[i]];

      if (!last_result.tag_match[i]) {
        e.tag = pc_tag;
        e.weight = 0;
        e.usefulness = 0;
        continue;
      }

      e.weight += t;

      if (correct && e.usefulness < 7)
        e.usefulness++;
      if (!correct && e.usefulness > 0)
        e.usefulness--;
    }

    adjust_threshold(correct);
  }
}

void hashed_perceptron_elite::adjust_threshold(bool correct)
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
