#include "idpn.h"

int idpn::clamp(int w) {
  if (w > MAX_WEIGHT) return MAX_WEIGHT;
  if (w < MIN_WEIGHT) return MIN_WEIGHT;
  return w;
}

// Proper 256-bit history folding hash (NO overflow)
std::size_t idpn::index(champsim::address ip) {

  std::size_t hash = ip.to<std::size_t>();

  for (int i = 0; i < HISTORY_LEN; i += 64) {

    std::uint64_t chunk = 0;

    for (int b = 0; b < 64 && (i + b) < HISTORY_LEN; b++) {
      if (ghistory[i + b])
        chunk |= (1ULL << b);
    }

    hash ^= chunk;
    hash = (hash << 13) ^ (hash >> 7); // extra mixing
  }

  return hash % NUM_BANKS;
}

// Compute neural score
int idpn::compute_sum(Bank& bank)
{
  int sum = bank.bias;

  for (int l = 0; l < NUM_LAYERS; l++) {
    for (int k = 0; k < KERNEL; k++) {

      int idx = k * dilations[l];
      if (idx >= HISTORY_LEN) break;

      int bit = ghistory[idx] ? 1 : -1;
      sum += bank.weights[l][k] * bit;
    }
  }

  return sum;
}

bool idpn::predict_branch(champsim::address ip)
{
  auto& bank = banks[index(ip)];
  int sum = compute_sum(bank);
  return sum >= 0;
}

void idpn::last_branch_result(champsim::address ip,
                              champsim::address,
                              bool taken,
                              uint8_t)
{
  auto& bank = banks[index(ip)];

  int sum = compute_sum(bank);
  bool prediction = sum >= 0;
  int target = taken ? 1 : -1;

  // Train only if wrong or weak confidence
  if (prediction != taken || std::abs(sum) < TRAIN_THRESHOLD) {

    bank.bias = clamp(bank.bias + target);

    for (int l = 0; l < NUM_LAYERS; l++) {
      for (int k = 0; k < KERNEL; k++) {

        int idx = k * dilations[l];
        if (idx >= HISTORY_LEN) break;

        int bit = ghistory[idx] ? 1 : -1;

        bank.weights[l][k] =
            clamp(bank.weights[l][k] + target * bit);
      }
    }
  }

  // Update global history
  ghistory <<= 1;
  ghistory[0] = taken;
}
