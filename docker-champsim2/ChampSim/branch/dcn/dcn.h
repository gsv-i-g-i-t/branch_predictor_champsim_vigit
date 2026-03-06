#ifndef DCN_BP
#define DCN_BP

#include <array>
#include <bitset>
#include <vector>
#include "modules.h"

struct dcn : champsim::modules::branch_predictor {

  static constexpr int HISTORY_LEN = 256;
  static constexpr int KERNEL_SIZE = 8;
  static constexpr int NUM_LAYERS = 4;
  static constexpr int MAX_WEIGHT = 127;
  static constexpr int MIN_WEIGHT = -128;
  static constexpr int THRESHOLD = 64;

  std::bitset<HISTORY_LEN> ghistory;

  // Dilations: 1,2,4,8
  std::array<int, NUM_LAYERS> dilations = {1,2,4,8};

  // Layer weights [layer][kernel]
  std::array<std::array<int, KERNEL_SIZE>, NUM_LAYERS> weights;

  // Layer biases
  std::array<int, NUM_LAYERS> biases;

  // Final combiner weights
  std::array<int, NUM_LAYERS> final_weights;
  int final_bias = 0;

  using branch_predictor::branch_predictor;

  bool predict_branch(champsim::address ip);
  void last_branch_result(champsim::address ip, champsim::address branch_target, bool taken, uint8_t branch_type);

  int clamp(int w);
};

#endif
