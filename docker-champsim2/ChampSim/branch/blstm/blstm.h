#ifndef BLSTM_BP
#define BLSTM_BP

#include <array>
#include <bitset>
#include <cmath>
#include <cstdint>
#include "modules.h"

struct blstm : champsim::modules::branch_predictor {

  static constexpr int IP_TABLE = 4096;
  static constexpr int HIDDEN = 16;

  // Per-branch hidden & cell state
  std::array<std::array<float, HIDDEN>, IP_TABLE> h{};
  std::array<std::array<float, HIDDEN>, IP_TABLE> c{};

  // LSTM weights
  std::array<std::array<float, HIDDEN>, HIDDEN> Wi{}, Wf{}, Wo_gate{}, Wg{};
  std::array<float, HIDDEN> bi{}, bf{}, bo{}, bg{};

  // Output layer
  std::array<float, HIDDEN> Wout{};

  using branch_predictor::branch_predictor;

  std::size_t ip_index(champsim::address ip);
  float sigmoid(float x);
  float tanh_clip(float x);
  float predict_score(champsim::address ip);
  bool predict_branch(champsim::address ip);
  void last_branch_result(champsim::address ip,
                          champsim::address,
                          bool taken,
                          uint8_t);
};

#endif
