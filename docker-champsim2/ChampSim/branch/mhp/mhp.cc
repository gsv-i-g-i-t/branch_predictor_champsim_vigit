#include "mhp.h"

inline int mhp::clamp(int x) {
  if (x > MAX_W) return MAX_W;
  if (x < MIN_W) return MIN_W;
  return x;
}

inline std::size_t mhp::index(champsim::address ip, int head_id)
{
  return (ip.to<uint64_t>() ^ (head_id * 0x9e3779b97f4a7c15ULL))
         & (PERCEPTRONS - 1);
}

int mhp::compute_sum(champsim::address ip)
{
  int sum = 0;

  for (int h = 0; h < NUM_HEADS; h++) {
    auto idx = index(ip, h);
    sum += heads[h].bias[idx];

    for (int i = 0; i < HIST_PER_HEAD[h]; i++) {
      int bit = ghistory[i] ? 1 : -1;
      sum += heads[h].weights[idx][i] * bit;
    }
  }

  return sum;
}

bool mhp::predict_branch(champsim::address ip)
{
  return compute_sum(ip) >= 0;
}

void mhp::last_branch_result(champsim::address ip,
                             champsim::address,
                             bool taken,
                             uint8_t)
{
  int sum = compute_sum(ip);
  bool pred = sum >= 0;
  int target = taken ? 1 : -1;

  int threshold = 1.93 * GHIST + 14;  // perceptron threshold heuristic

  if (pred != taken || std::abs(sum) < threshold) {

    for (int h = 0; h < NUM_HEADS; h++) {
      auto idx = index(ip, h);

      heads[h].bias[idx] =
          clamp(heads[h].bias[idx] + target);

      for (int i = 0; i < HIST_PER_HEAD[h]; i++) {
        int bit = ghistory[i] ? 1 : -1;
        heads[h].weights[idx][i] =
            clamp(heads[h].weights[idx][i] + target * bit);
      }
    }
  }

  ghistory <<= 1;
  ghistory[0] = taken;
}
