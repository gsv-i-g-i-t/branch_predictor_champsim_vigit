#ifndef IDPN_BP
#define IDPN_BP

#include <array>
#include <bitset>
#include <cstdint>
#include <cmath>
#include "modules.h"

struct idpn : champsim::modules::branch_predictor {

  static constexpr int HISTORY_LEN = 256;
  static constexpr int NUM_BANKS = 2048;      // larger bank = less interference
  static constexpr int NUM_LAYERS = 5;
  static constexpr int KERNEL = 16;

  static constexpr int MAX_WEIGHT = 127;
  static constexpr int MIN_WEIGHT = -128;
  static constexpr int TRAIN_THRESHOLD = 64;

  std::bitset<HISTORY_LEN> ghistory;

  std::array<int, NUM_LAYERS> dilations = {1,2,4,8,16};

  struct Bank {
    int bias = 0;
    std::array<std::array<int, KERNEL>, NUM_LAYERS> weights{};
  };

  std::array<Bank, NUM_BANKS> banks;

  using branch_predictor::branch_predictor;

  int clamp(int w);
  std::size_t index(champsim::address ip);
  int compute_sum(Bank& bank);
  bool predict_branch(champsim::address ip);
  void last_branch_result(champsim::address ip,
                          champsim::address target,
                          bool taken,
                          uint8_t branch_type);
};

#endif
