#include "hashed_perceptron_v2.h"

bool hashed_perceptron_v2::predict_branch(champsim::address pc)
{
  auto pc_slice = pc.slice_lower<TABLE_INDEX_BITS>().to<uint64_t>();

  perceptron_result result;

  // ---- Bias table (table 0)
  result.indices[0] = pc_slice & (TABLE_SIZE - 1);

  // ---- Geometric history tables
  for (std::size_t i = 0; i < ghist_words.size(); i++) {
    uint64_t h = ghist_words[i].value();

    // skewed indexing to reduce aliasing
    uint64_t idx =
        (h * (0x9e3779b97f4a7c15ULL + i)) ^
        (pc_slice * (0xC2B2AE3D27D4EB4FULL + i));

    result.indices[i+1] = idx & (TABLE_SIZE - 1);
  }

  // compute perceptron output
  result.yout = 0;
  for (std::size_t i = 0; i < NTABLES; i++)
    result.yout += tables[i][result.indices[i]].value();

  last_result = result;

  return result.yout >= THRESHOLD;
}

void hashed_perceptron_v2::last_branch_result(champsim::address pc,
                                              champsim::address,
                                              bool taken,
                                              uint8_t)
{
  auto pc_bit = pc.slice_lower<champsim::data::bits{1}>().to<uint64_t>();

  // ---- Path-based history update (direction XOR PC bit)
  for (auto& hist : ghist_words) {
    bool input = taken ^ pc_bit;
    hist.push_back(input);
  }

  bool prediction_correct =
      (taken == (last_result.yout >= THRESHOLD));

  // stronger weak margin
  bool prediction_weak =
      (std::abs(last_result.yout) < theta * 2);

  if (!prediction_correct || prediction_weak) {

    for (std::size_t i = 0; i < NTABLES; i++)
      tables[i][last_result.indices[i]] += taken ? 1 : -1;

    adjust_threshold(prediction_correct);
  }
}

void hashed_perceptron_v2::adjust_threshold(bool correct)
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
