#ifndef MHP_BP
#define MHP_BP

#include <array>
#include <bitset>
#include <cstdint>
#include <cmath>
#include "modules.h"

struct mhp : champsim::modules::branch_predictor {

  static constexpr int GHIST = 256;
  static constexpr int NUM_HEADS = 4;
  static constexpr int PERCEPTRONS = 2048;
  static constexpr int HIST_PER_HEAD[NUM_HEADS] = {32, 64, 128, 256};

  static constexpr int MAX_W = 63;
  static constexpr int MIN_W = -64;

  std::bitset<GHIST> ghistory;

  struct Head {
    std::array<std::array<int8_t, GHIST>, PERCEPTRONS> weights{};
    std::array<int8_t, PERCEPTRONS> bias{};
  };

  std::array<Head, NUM_HEADS> heads;

  using branch_predictor::branch_predictor;

  inline int clamp(int x);
  inline std::size_t index(champsim::address ip, int head_id);
  int compute_sum(champsim::address ip);
  bool predict_branch(champsim::address ip);
  void last_branch_result(champsim::address ip,
                          champsim::address,
                          bool taken,
                          uint8_t);
};

#endif
